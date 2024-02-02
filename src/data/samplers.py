"""
    Variety of sampling mechanisms
"""

import time
import math
import itertools
from itertools import accumulate, repeat, chain
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from sklearn.covariance import LedoitWolf
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, value
from pulp.apis import PULP_CBC_CMD

from .utils import collate
from .utils import convert_to_graphs, convert_to_item
from model import get_model_cls


class DatasetSampler:
    def __init__(self, args, dataset, visit_counts=None, **kwargs):
        self.args = args
        self.dataset = dataset
        self.num_vars = dataset.num_vars
        self.reset()  # initialize variables
        if visit_counts is not None:
            self.visit_counts = visit_counts

    def compute_scores(self):
        raise Exception("implement me")

    def sample_nodes(self, num_nodes):
        """
            @param (np.ndarray) scores  NxN
        """
        scores = self.compute_scores()
        joint_scores = scores * 1 / np.sqrt(1 + self.visit_counts)
        sampled = []
        for i in range(num_nodes):
            sampled_arr = np.array(sampled, dtype=int)
            if len(sampled) == 0:
                p = joint_scores.sum(0)
            else:
                p = joint_scores[sampled_arr]
                p[:,sampled_arr] = 0
                p = p.sum(0)
            # remove sampled from consideration
            p = p / p.sum()  # normalize probabilities to 1
            v = np.random.choice(len(p), (1,), p=p).item()
            sampled.append(v)
        # update visit counts
        update_counts = np.zeros((self.num_vars, self.num_vars))
        sampled = np.array(sampled)
        edges = cartesian_prod([sampled, sampled])
        update_counts[edges[:,0], edges[:,1]] = 1
        self.visit_counts += update_counts
        assert len(sampled) == len(set(sampled)) == num_nodes
        return sampled

    def sample_batches(self):
        raise Exception("implement me")

    def callback(self, *args, **kwargs):
        # optionally called after sampling every batch
        return

    def reset(self):
        self.visit_counts = np.zeros((self.num_vars, self.num_vars))
        # apparently catting to the empty tensor works like a charm
        self.graphs = torch.empty((0,), dtype=torch.long)
        self.orders = torch.empty((0,), dtype=torch.long)


class ObservationalSampler(DatasetSampler):
    """
        Sampler for observational data
    """
    def sample_batches(self, num_batches, batch_size, num_vars_batch):
        """
            Called by Datasets
        """
        batches = []
        for i in range(num_batches):
            # sample dataset points
            idxs = np.random.choice(len(self.dataset), (batch_size,),
                    replace=False)
            # sample nodes
            nodes = self.sample_nodes(num_vars_batch)
            # broadcast axis
            batch = self.dataset.data[idxs[:, np.newaxis], nodes]
            batches.append((batch, nodes))
            self.callback([(batch, nodes)])  # for learned sampler
        # compute global features based on last batch
        feats = compute_features(self.dataset.data[idxs].T)
        return batches, feats


class InterventionalSampler(DatasetSampler):
    """
        Sampler for interventional data
    """
    def sample_batches(self, num_batches, batch_size, num_vars_batch):
        """
            Called by Datasets
        """
        batches = []
        # +1 for observational
        points_per_env = batch_size // (num_vars_batch + 1)
        for i in range(num_batches):
            # sample nodes
            nodes = self.sample_nodes(num_vars_batch)
            # sample regimes that impact those points
            reg_idx = []
            for v in nodes:
                reg_idx.extend(self.dataset.node_to_regime[v])
            # for Sachs
            if len(reg_idx) < num_vars_batch:
                reg_idx = sorted(set(reg_idx))
                for _ in range(num_vars_batch - len(reg_idx)):
                    reg_idx.append(0)  # observational
            else:
                reg_idx = np.random.choice(sorted(set(reg_idx)),
                                           num_vars_batch,
                                           replace=False)
            # sample examples from each regime
            batch = []
            for reg in reg_idx:
                # sample dataset points
                idxs = np.random.choice(self.dataset.regimes[reg],
                                        points_per_env,
                                        replace=False)[:, np.newaxis]
            batch.append(self.dataset.data[idxs, nodes])
            # add observational
            idxs = np.random.choice(self.dataset.regimes[0],
                                    points_per_env,
                                    replace=False)[:, np.newaxis]
            batch.append(self.dataset.data[idxs, nodes])
            batch = np.stack(batch, axis=0)

            # re-number nodes in regimes to local batch idx
            # and remove intervened nodes outside of sampled set
            node_renumber = {node:i for i, node in enumerate(nodes)}
            regimes = [self.dataset.idx_to_regime[reg] for reg in reg_idx]
            regimes = [[node_renumber.get(x) for x in reg] for reg in regimes]
            regimes = [[x for x in reg if x is not None] for reg in regimes]
            regimes.append([])

            batches.append((batch, nodes, regimes))
            self.callback([(batch, nodes, regimes)])  # for learned sampler

        idxs = np.random.choice(len(self.dataset), (batch_size,),
                replace=False)
        feats = compute_features(self.dataset.data[idxs].T)

        return batches, feats


class RandomSampler(DatasetSampler):
    def compute_scores(self):
        scores = np.ones((self.num_vars, self.num_vars))
        scores = scores * (1 - np.eye(len(scores)))  # zero out diagonal
        return scores


class CorrelationSampler(DatasetSampler):
    def compute_scores(self):
        idxs = np.random.choice(len(self.dataset),
                (self.args.fci_batch_size,), replace=False)
        batch_score = self.dataset[idxs]
        score = compute_features(batch_score.T)
        score = score * (1 - np.eye(len(score)))  # zero out diagonal
        score = np.abs(score)  # make positive
        return score


class LearnedSampler(DatasetSampler):
    """
        Sampler with learned Selector model
    """
    def __init__(self, args, dataset, visit_counts=None,
                 model=None, run_alg=None):
        """
            @param (nn.Module)  model
        """
        super().__init__(args, dataset, visit_counts=visit_counts)
        model_cls = get_model_cls(args)
        self.model = model_cls.load_from_checkpoint(args.checkpoint_path)
        self.model.eval()
        self.run_alg = run_alg
        self.reset()  # initialize graphs and orders

    def compute_scores(self, block_size=10):
        # initialize to random selection
        if len(self.graphs) < block_size:
            scores = np.ones((self.num_vars, self.num_vars))
            scores = scores * (1 - np.eye(len(scores)))  # zero out diagonal
            return scores

        # otherwise only run model every 10 batches for speed
        if len(self.graphs) % block_size != 0:
            return self.scores

        # sample features
        idxs = np.random.choice(len(self.dataset),
                self.args.fci_batch_size, replace=False)
        feats = compute_features(self.dataset.data[idxs].T)

        # convert_to_item will ASSIGN the index based on orders
        # so this changes per-batch
        item = convert_to_item(self.dataset,
                               feats, self.graphs, self.orders)
        # run existing samples through model
        batch = collate(self.args, [item])  # singleton
        # no need for gradients and set .eval() mode
        with torch.no_grad():
            output = self.model.encoder(batch)
            pred, true = self.model.symmetrize(output, batch, reduce=False)
            pred, true = pred[0], true[0]  # only 1 by construction
            # select P(forward) and P(backward), remove P(no edge)
            # corresponding to dim=1, dim=2, and dim=0
            pred = F.softmax(pred, dim=-1)[:,1:].t()  # (2, (N-1)^2/2)
            pred = entropy(pred.numpy(), axis=0)
            # coerce back into 2D array
            scores = np.zeros((self.num_vars, self.num_vars))
            mask = np.tri(self.num_vars, k=-1, dtype=bool)
            scores[mask] = pred
            scores = scores + scores.T
        # store for next 10 batches
        self.scores = scores
        # reset visit counts
        self.visit_counts = np.zeros((self.num_vars, self.num_vars))
        return scores

    # NOTE: this is the ILP version, which is MUCH slower than greedy
    # This was NOT used in any of our results
    #def sample_nodes(self, num_nodes):
    #    """
    #        @param (np.ndarray) scores  NxN
    #    """
    #    #start = time.time()
    #    scores = self.compute_scores()
    #    #end = time.time()
    #    #print(end - start, "score")
    #    # >>> try without visit counts first
    #    joint_scores = scores * 1 / np.sqrt(1 + self.visit_counts)
    #    # <<<
    #    #start = time.time()
    #    with open("dummy", "w") as f:
    #        with redirect_stdout(f):
    #            sampled = ilp_max_k_node_subgraph(scores,
    #                                              K=self.args.fci_vars)
    #    #end = time.time()
    #    #print(end - start, "ilp")
    #    # update visit counts
    #    update_counts = np.zeros((self.num_vars, self.num_vars))
    #    sampled = np.array(sampled)
    #    edges = cartesian_prod([sampled, sampled])
    #    update_counts[edges[:,0], edges[:,1]] = 1
    #    self.visit_counts += update_counts
    #    assert len(sampled) == len(set(sampled)) == num_nodes
    #    return sampled

    def callback(self, next_batch):
        """
            Runs algorithm on next batch and adds to cache
        """
        results = self.run_alg(next_batch)
        graphs, orders = convert_to_graphs(results,
                                           self.dataset)
        self.graphs = torch.cat([self.graphs, graphs], dim=0)
        self.orders = torch.cat([self.orders, orders], dim=0)


def ilp_max_k_node_subgraph(scores, K=5):
    """
    Written with the help of ChatGPT so apologies if this is suboptimal.
    This selects MAXIMUM score subset.
    """

    num_nodes = scores.shape[0]

    # Create the ILP problem
    prob = LpProblem("MaxKNodeSubgraph", sense=LpMaximize)

    # Create binary variables for each node indicating whether it's in the subset
    nodes = range(num_nodes)
    x = LpVariable.dicts("Node", nodes, cat="Binary")

    # Create binary variables for each edge indicating whether it's selected
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    y = LpVariable.dicts("Edge", edges, cat="Binary")

    # Objective function: maximize the total edge weight within the subset
    prob += lpSum(scores[i, j] * y[(i, j)] for i in range(num_nodes) for j in range(num_nodes) if i != j)

    # Constraint: the subset should have exactly K nodes
    prob += lpSum(x[node] for node in nodes) == K

    # Constraint: if x[i] = 1 is selected AND x[j] = 1 is selected, then y[(i, j)] = 1
    # Otherwise, if either x[i] = 0 OR x[j] = 0, then y[(i, j)] = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                prob += y[(i, j)] >= x[i] + x[j] - 1
                prob += y[(i, j)] <= x[i]
                prob += y[(i, j)] <= x[j]

    # Solve the ILP problem
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract the solution
    selected_nodes = [node for node, var in x.items() if value(var) == 1]

    return selected_nodes


def compute_features(x):
    """
    Implementation depends on matrix size
    x: (num_vars, num_samples)
    """
    if x.shape[0] < 100:
        return np.linalg.pinv(np.cov(x), rcond=1e-10)
    lw = LedoitWolf()
    lw.fit(x.T)
    invcovs = lw.get_precision()
    return invcovs


def cartesian_prod(arrays):
    """
        https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/49445693#49445693
    """
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)

