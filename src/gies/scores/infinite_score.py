# Copyright 2022 Olga Kolotuhina, Juan L. Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Function to run GIES with the modified score
# (scores.exp_gauss_int_l0_pen) which does not remove the term
# depending on the intervention targets; the original implementation
# for GIES does this because in their setting the intervention targets
# are fixed


"""Implements a population version of the l0-penalized interventional Gaussian score"""

import numpy as np
from .decomposable_score import DecomposableScore
from .experimental import _regress


class InfiniteScore(DecomposableScore):
    def __init__(
        self,
        cov,
        interv,
        mean=[],
        sigma=[],
        K=[],
        lmbda=None,
        method="scatter",
        cache=True,
        debug=0,
    ):
        super().__init__(cov, interv, cache=cache, debug=debug)
        self.p = cov[0].shape[1]
        self.n_obs = np.ones(len(interv))
        self.total_num_interv = sum(len(inter) for inter in interv)
        self.sample_cov = cov
        self.N = sum(self.n_obs)
        self.lmbda = 0
        self.num_not_interv = np.zeros(self.p)
        self.part_sample_cov = np.zeros((self.p, self.p, self.p))
        self.C = 0
        if mean:
            for i in range(len(self.interv)):
                for k in self.interv[i]:
                    self.C += 0.5 * (
                        np.log(K[i][k, k]) - self.sample_cov[i][k, k] * K[i][k, k]
                    )
                mask = np.zeros_like(sigma[i])
                mask[np.ix_(interv[i], interv[i])] = 1
                self.C += 0.5 * mask @ mean[i] @ K[i] @ mean[i]

        # Computing the numbers of non-interventions of a variable and the corresponding partial covariance matrix
        for k in range(self.p):
            for (i, n) in enumerate(self.n_obs):
                if k not in set(self.interv[i]):
                    self.num_not_interv[k] += n
                    self.part_sample_cov[k] += self.sample_cov[i] * n
            self.part_sample_cov[k] = self.part_sample_cov[k] / self.num_not_interv[k]

    def full_score(self, A):
        """
        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a sample from a single environment, by finding the maximum
        likelihood estimates of the corresponding connectivity matrix
        (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        # Compute MLE
        B, omegas = self._mle_full(A)
        likelihood = 0
        for j, sigma in enumerate(self.part_sample_cov):
            likelihood += -self.num_not_interv[j] * (1 + np.log(omegas[j]))
        l0_term = self.lmbda * (np.sum(A != 0) + self.p)
        score = 0.5 * likelihood - l0_term + self.C
        return score

    # Note: self.local_score(...), with cache logic, already defined
    # in parent class DecomposableScore.

    def _compute_local_score(self, k, pa):
        """
        Given a node and its parents, return the local l0-penalized
        log-likelihood of a sample from a single environment, by finding
        the maximum likelihood estimates of the weights and noise term
        variances.

        Parameters
        ----------
        x : int
            a node.
        pa : set of ints
            the node's parents.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        pa = list(pa)
        # Compute MLE
        b, sigma = self._mle_local(k, pa)
        # Compute log-likelihood (without log(2π) term)
        n = self.num_not_interv[k]
        likelihood = -0.5 * n * (1 + np.log(sigma))
        #  Note: the number of parameters is the number of parents (one
        #  weight for each) + the marginal variance of x
        l0_term = self.lmbda * (len(pa) + 1)
        score = likelihood - l0_term
        return score

    # --------------------------------------------------------------------
    #  Functions for the maximum likelihood estimation of the
    #  weights/variances

    def _mle_full(self, A):
        """
        Finds the maximum likelihood estimate for the whole graph,
        specified by the adjacency A.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        B : np.array
            the connectivity (weights) matrix, which respects the
            adjacency in A.
        omegas : np.array
            the estimated noise-term variances of the observed
            variables.

        """
        B = np.zeros(A.shape)
        omegas = np.zeros(self.p)
        for j in range(self.p):
            parents = np.where(A[:, j] != 0)[0]
            B[:, j], omegas[j] = self._mle_local(j, parents)
        return B, omegas

    def _mle_local(self, k, pa):
        """Finds the maximum likelihood estimate of the local model
        between a node and its parents.

        Parameters
        ----------
        x : int
            a node.
        pa : set of ints
            the node's parents.

        Returns
        -------
        b : np.array
            an array of size p, with the estimated weights from the
            parents to the node, and zeros for non-parent variables.
        sigma : float
            the estimate noise-term variance of variable x

        """
        pa = list(pa)
        b = np.zeros(self.p)
        S_k = self.part_sample_cov[k]
        S_kk = S_k[k, k]
        S_pa_k = S_k[pa, :][:, k]
        b[pa] = _regress(k, pa, S_k)
        sigma = S_kk - b[pa] @ S_pa_k
        return b, sigma
