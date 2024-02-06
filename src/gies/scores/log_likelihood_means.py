# Copyright 2022 Juan L. Gamella

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

"""Module containing the code to compute the log-likelihood of a
model (connectivity, noise term means/variances) from data of a single
or multiple environments.
"""

import numpy as np

# --------------------------------------------------------------------
# Functions to compute the full/local log-likelihood


def full(B, noise_means, noise_variances, XX):
    p = len(B)
    likelihood = 0
    for e, X in enumerate(XX):
        inv_omega = np.diag(1 / noise_variances[e])
        inv_cov = (np.eye(p) - B.T).T @ inv_omega @ (np.eye(p) - B.T)
        log_term = -len(X) * np.log(1 / noise_variances[e].prod())
        mu = np.linalg.inv(np.eye(p) - B.T) @ noise_means[e]
        emp_cov = corr(X - mu)
        trace_term = np.trace(emp_cov @ inv_cov)
        # print(log_term, trace_term)
        likelihood += log_term + trace_term
    return -0.5 * likelihood


def local(j, b, noise_means, noise_variances, XX):
    return local_raw(j, b, noise_means, noise_variances, XX)


def full_raw(B, noise_means, noise_variances, XX):
    """Compute the log-likelihood of the given weights (i.e. B), noise
    term means and variances.

    Parameters
    ----------
    B : np.array
        The connectivity matrix.
    noise_means : np.array
        The means of the noise terms of each variable.
    noise_variances : np.array
        The noise term variances of each variable.
    XX : list(np.array)
        The complete joint observations for each environment, where
        each element of the list is an np.array of size n_e x p, where
        p is the total number of variables (same size as b) and n_e is
        the number of observations from that environment.

    Returns
    -------
    likelihood : float
        The log-likelihood of the given parameters

    """
    p = len(B)
    likelihood = 0
    for e, X in enumerate(XX):
        inv_omega = np.diag(1 / noise_variances[e])
        inv_cov = (np.eye(p) - B.T).T @ inv_omega @ (np.eye(p) - B.T)
        log_term = -len(X) * np.log(1 / noise_variances[e].prod())
        # In tests, both ways of computing the log term seemed equally
        # fast, for p=5, n=10K
        # log_term = -len(X) * np.log(np.linalg.det(inv_cov))
        # assert np.isclose(log_term, -len(X) * np.log(np.linalg.det(inv_cov)))
        aux = X - np.linalg.inv(np.eye(p) - B.T) @ noise_means[e]
        cov_term = 0
        for i, x in enumerate(aux):
            cov_term += x @ inv_cov @ x
        likelihood += log_term + cov_term
        # print(log_term, cov_term)
        # In tests, using the trace was almost 4x slower for p=5, n=10K
        # trace_term = np.trace(aux @ inv_cov @ aux.T)
        # likelihood += log_term + trace_term
    return -0.5 * likelihood


def local_raw(j, b, noise_means, noise_variances, XX):
    """Compute the conditional likelihood of the marginal observations of
    a variable given observations of its hypothetical parents, given
    the weights (b) for those parents and the variable's error term
    mean and variance over each environment.

    Parameters
    ----------
    j : int
        The variable's index.
    b : np.array
        A p-sized array with the weights for the parents of j and 0s
        elswhere.
    noise_means : np.array
        A one-dimensional array with the mean of j's noise term for each
        environment.
    noise_variances : np.array
        A one-dimensional array with the variance of j's noise term for each
        environment.
    XX : list(np.array)
        The complete joint observations for each environment, where
        each element of the list is an np.array of size n_e x p, where
        p is the total number of variables (same size as b) and n_e is
        the number of observations from that environment.

    Returns
    -------
    likelihood : float
        The resulting log-likelihood.

    """
    likelihood = 0
    # Iterate over environments
    for e, X in enumerate(XX):
        log_term = len(X) * np.log(noise_variances[e])
        aux = X[:, j] - X @ b - noise_means[e]
        sum_term = aux @ aux / noise_variances[e]
        likelihood += -0.5 * (log_term + sum_term)
    return likelihood


def corr(X, scale=True):
    """Return the empirical correlation matrix given observations from a
    random vector X.

    Parameters
    ----------
    X : array_like
        The observations of the random vector, with columns
        corresponding to variables.
    normalize : bool
        If the resulting matrix should be scaled by the number of
        observations. Defaults to True.

    Returns
    -------
    corr : np.ndarray
        The correlation matrix.

    """
    corr = np.dot(X.T, X) if scale else np.dot(X.T, X) / len(X)
    return corr
