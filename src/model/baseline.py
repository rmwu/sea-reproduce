import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryAccuracy


class CausalBaseline(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_vars = args.num_vars

        # validation meters
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()

        self.save_hyperparameters()

    def forward(self, batch):
        """
        Used on predict_dataloader
        """
        start = time.time()  # keep track of GPU time
        output = self.encode(batch)
        end = time.time()  # keep track of GPU time
        results = self.compute_metrics_per_graph(output, batch,
                save_preds=True)
        # run with batch_size=1 for accurate timing
        results["time"] = batch["time"] + (end - start)
        return results

    def scatter_tokens(self, batch):
        """
            Scatter x from [B, T, #sampled] to [B, T, #unique]
                                    =k*k                 =K
        """
        x = batch["input"]  # [B, T, k*k]
        batch_size, num_trials, _ = x.size()
        num_edges = batch["index"].max() + 1  # [B, K]
        x_expand = torch.zeros(batch_size, num_trials, num_edges,
                device=x.device, dtype=x.dtype)
        index = batch["index"]  # [B, T, k*k]
        assert x.size() == index.size()
        # [B, T, k*k] -> [B, T, K]
        x_expand = x_expand.scatter_(dim=2, index=index, src=x)
        return x_expand

    def encode(self, batch):
        """
            Aggregate based on heuristics
        """
        x_expand = self.scatter_tokens(batch)
        # BE CAREFUL HERE
        # by design, I always set forward edge to "2"
        edge_pred = (x_expand == 2).float()
        # collapse over time
        edge_pred = edge_pred.mean(1)
        return edge_pred

    def compute_metrics_per_graph(self, output, batch, save_preds=False):
        """
            Split up individual graphs from batch
        """
        pred = output[0]
        true = batch["label"][0]
        unique = batch["unique"][0] - 1  # remove padding
        true = true[unique[:,0], unique[:,1]]

        # symmetrize
        batch.update(get_mask(batch))
        forward_mask = batch["forward_mask"]
        backward_mask = batch["backward_mask"]
        backward_order = [order for order in batch["backward_order"]]

        auroc, auprc, acc = [], [], []
        if save_preds:
            pred_list, true_list = [], []
        # symmetrize edges
        p_forward = pred[forward_mask]
        t_forward = true[forward_mask]
        p_backward = pred[backward_mask][backward_order]
        t_backward = true[backward_mask][backward_order]
        p = torch.cat([p_forward, p_backward], dim=-1)
        t = torch.cat([t_forward, t_backward], dim=0)
        assert p.shape == t.shape
        # convert prediction to list
        auroc.append(self.auroc(p, t).item())
        auprc.append(self.auprc(p, t).item())
        acc.append(self.acc(p, t).item())
        if save_preds:
            pred_list.append(p.tolist())
            true_list.append(t.tolist())

        outputs = {}
        outputs["auroc"] = auroc
        outputs["auprc"] = auprc
        outputs["acc"] = acc
        # need to save these...
        outputs["key"] = batch["key"]
        if save_preds:
            outputs["pred"] = pred_list
            outputs["true"] = true_list
        return outputs


def get_mask(batch):
    unique = batch["unique"][0] - 1  # subtract padding + batch dim
    direction = (unique[:,0] < unique[:,1])  # diagonal not in unique
    rev_direction = (unique[:,0] > unique[:,1])
    unique_u = torch.flip(unique[rev_direction], dims=[1])
    # re-order unique_u to match unique
    # because e.g. (1,2) (1,4) (2,1) (2,3) (3,2) (4,1)
    # flipped to   (1,2) (1,4) (1,2)* (2,3) (2,3)* (1,4)*
    # ends up out of order from (1,2) (1,4) (2,3)
    outer_sort = torch.argsort(unique_u[:,1])
    inner_sort = torch.sort(unique_u[outer_sort,0], stable=True)[1]
    order = torch.as_tensor(outer_sort[inner_sort])
    item = {
        "forward_mask": direction,
        "backward_mask": rev_direction,
        "backward_order": order
    }
    return item

