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

"""
"""

import numpy as np
from .decomposable_score import DecomposableScore

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment


class GaussIntL0Pen(DecomposableScore):
    """
    Implements a cached l0-penalized gaussian likelihood score for the GIES setting.

    """

    def __init__(self, data, interv, lmbda=None, cache=True, debug=0):
        """Creates a new instance of the class.

        Parameters
        ----------
        data : list of numpy.ndarray
            every matrix in the list corresponds to an environment,
            the nxp matrix containing the observations of each
            variable (each column corresponds to a variable).
        interv: a list of lists
            a list of the interventions sets which
            corresponds to the environments in data
        lmbda : float or NoneType, optional
            the regularization parameter. If None, defaults to the BIC
            score, i.e. lmbda = 1/2 * log(n), where n is the number of
            observations.
        cache : bool, optional
           if computations of the local score should be cached for
           future calls. Defaults to True.
        debug : int, optional
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity.

        """
        super().__init__(data, interv, cache=cache, debug=debug)
        self.p = self._data[0].shape[1]
        self.n_obs = np.array([len(env) for env in self._data])
        # Computing the sample covariances
        self._data = [sample - sample.mean(axis=0) for sample in self._data]
        self.sample_cov = np.array(
            [1 / self.n_obs[ind] * env.T @ env for (ind, env) in enumerate(self._data)]
        )
        # Discarded options for computing the sample covariances
        #  a) This is different to how it is computed in the PCALG pacakge
        # self.sample_cov = np.array([np.cov(env, rowvar=False, ddof=0) for env in self._data])
        #  c) This also sometimes yields outputs which are different to what is computed in PCALG
        # sample_cov = []
        # for i, env in enumerate(self._data):
        #     mean = np.mean(env, axis=0)
        #     aux = env - mean
        #     sample_cov.append(1 / self.n_obs[i] * aux.T @ aux)
        # self.sample_cov = np.array(sample_cov)

        self.N = sum(self.n_obs)
        self.lmbda = 0.5 * np.log(self.N) if lmbda is None else lmbda
        self.num_not_interv = np.zeros(self.p)
        self.part_sample_cov = np.zeros((self.p, self.p, self.p))

        # Check that the interventions form a conservative family of targets
        for j in range(self.p):
            if sum(i.count(j) for i in self.interv) == len(self._data):
                raise ValueError("The family of targets is not conservative")

        # Computing the numbers of non-interventions of a variable and the corresponding partial covariance matrix
        # this seems highly parallelizable for optimization...
        for k in range(self.p):
            # >>> buffer
            sample_covs = []
            ns = []
            # <<< buffer
            for (i, n) in enumerate(self.n_obs):
                if k not in set(self.interv[i]):
                    self.num_not_interv[k] += n
                    #self.part_sample_cov[k] += self.sample_cov[i] * n
                    sample_covs.append(self.sample_cov[i])
                    ns.append(n)
            #self.part_sample_cov[k] = self.part_sample_cov[k] / self.num_not_interv[k]
            if len(sample_covs) > 0:
                ns = np.array(ns).reshape(len(ns), 1, 1)
                sample_covs = (np.array(sample_covs) * ns /
                        self.num_not_interv[k]).sum(0)
                self.part_sample_cov[k] = sample_covs
            else:
                # >>> this should never be run lol
                print(self.num_not_interv)
            # <<<

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
            gamma = 1 / omegas[j]
            likelihood += self.num_not_interv[j] * (
                np.log(gamma)
                - gamma
                * (np.eye(self.p) - B)[:, j]
                @ sigma
                @ (np.eye(self.p) - B)[:, j].T
            )
        l0_term = self.lmbda * (np.sum(A != 0) + self.p)
        score = 0.5 * likelihood - l0_term
        #print(omegas, score)
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
        # >>> numerical stability fix
        log_sigma = np.where(sigma > 1e-6, sigma, -10)
        log_sigma = np.log(log_sigma, out=log_sigma, where=log_sigma > 0)
        # <<<
        likelihood = -0.5 * n * (1 + log_sigma)
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
        omegas = omegas + 1e-10
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


def _regress(j, pa, cov):
    # compute the regression coefficients from the
    # empirical covariance (scatter) matrix i.e. b =
    # Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
    try:
        return np.linalg.solve(cov[pa, :][:, pa], cov[j, pa])
    except:
        return np.linalg.lstsq(cov[pa, :][:, pa], cov[j, pa], rcond=1e-10)[0]

