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

"""Module containing the code to compute the log-likelihood of a model
(connectivity and variances) from data of a single or multiple
environments. This module is different to log_likelihood_means in that
the means are always set to their MLE. This offers the possibility of
using precomputed sample covariance matrices (see full and local
functions).
"""

import numpy as np

# --------------------------------------------------------------------
# Functions to compute the full/local log-likelihood


def full(B, omegas, sample_covariances, n_obs):
    """
    Compute the log-likelihood of the given weights (B) and noise
    variances (omegas).

    Parameters
    ----------
    B : np.array
        the connectivity matrix
    omegas : np.array
        the noise variances of each variable
    sample_covariances : list(np.array())
        the list of sample covariances (see np.cov) of the sample from
        each environment
    n_obs : list of ints
       the number of observations available from each environment
       (i.e. the sample size)

    Returns
    -------
    likelihood : float
        the log-likelihood of the given parameters

    """
    p = len(B)
    likelihood = 0
    for j, sigma in enumerate(sample_covariances):
        K = np.diag(1 / omegas[j])
        det_K = np.prod(1 / omegas[j])
        likelihood += n_obs[j] * (
            np.log(det_K)
            - np.trace(K @ (np.eye(p) - B.T) @ sigma @ (np.eye(p) - B.T).T)
        )
    return likelihood * 0.5


def local(j, b, omegas, sample_covariances, n_obs):
    """
    Compute the conditional likelihood of the marginal observations of
    a variable given observations of its hypothetical parents, given
    the weights (b) for those parents and the variable's error term
    variances over each environment (omegas).

    Parameters
    ----------
    j : int
        the variable's index
    b : np.array
        p-sized array with the weights for the parents of j and 0s
        elswhere
    omegas : np.array
        1d array with the noise term variance of j for each
        environment
    sample_covariances : list(np.array())
        the list of sample covariances (see np.cov) of the sample from
        each environment
    n_obs : list of ints
       the number of observations available from each environment
       (i.e. the sample size)

    Returns
    -------
    log_likelihood : float
        the resulting log_likelihood

    """
    likelihood = 0
    I = np.zeros_like(b)
    I[j] = 1
    for e, sigma in enumerate(sample_covariances):
        K = 1 / omegas[e]
        likelihood += n_obs[e] * (np.log(K) - K * (I - b) @ sigma @ (I - b))
    return likelihood * 0.5


def full_raw(B, omegas, data):
    """
    Compute the log likelihood of the given connectivity matrix (B) and noise
    variances (omegas).

    Parameters
    ----------
    B : np.array
        the connectivity matrix, where B[i,j] = w iff i -> j with weight w
    omegas : np.array
        the noise variances of each variable
    data : list(np.array)
        A list of samples from each environment, where each sample is
        an np.array with each column corresponding to a variable

    Returns
    -------
    likelihood : float
        the log-likelihood of the given parameters

    """
    sample_covariances = [np.cov(env, rowvar=False, ddof=0) for env in data]
    n_obs = [len(env) for env in data]  # number of observations/environment
    return full(B, omegas, sample_covariances, n_obs)


def local_raw(j, b, omegas, data):
    """
    Compute the conditional likelihood of the marginal observations of
    a variable given observations of its hypothetical parents, given
    the weights (b) for those parents and the variable's error term
    variances over each environment (omegas).

    Parameters
    ----------
    j : int
        the variable's index
    b : np.array
        p-sized array with the weights for the parents of j and 0s
        elswhere
    omegas : np.array
        1d array with the noise term variance of j for each
        environment
    data : list(np.array)
        the complete joint observations for each environment, where
         each element of the list is an np.array of size n_e x p,
         where p is the total number of variables (same size as b) and
         n_e is the number of observations from that environment.

    Returns
    -------
    log_likelihood : float
        the resulting log_likelihood

    """
    sample_covariances = [np.cov(env, rowvar=False, ddof=0) for env in data]
    n_obs = [len(env) for env in data]
    return local(j, b, omegas, sample_covariances, n_obs)


# def local(j,b,omegas,data):
#     """New with manually computed scatter matrix"""
#     data = data[0]
#     n = len(data)
#     normalized = np.hstack([data, np.ones((n,1))])
#     normalized -= normalized.mean(axis=0)
#     cov = normalized.T @ normalized / n
#     pa = np.where(b != 0)[0]
#     if len(pa) > 0:
#         sub_cov = cov[:,pa][pa,:]
#         b = cov[j,pa]
#         sigma = cov[j,j] - b @ np.linalg.solve(sub_cov, b)
#         #likelihood = -n * np.log(2*np.pi) - n * np.log(sigma) - 1 / sigma
#     else:
#         sigma = cov[j,j]
#     n = len(data)
#     likelihood = -0.5*n*( 1 + np.log(sigma/n))
#     return likelihood

# def local(j,b,omegas,data):
#     """From raw data"""
#     data = data[0]
#     n,p = data.shape
#     data = np.hstack([data, np.ones((n,1))])
#     pa = list(np.where(b != 0)[0]) + [p]
#     Y = data[:,j]
#     X = np.atleast_2d(data[:,pa])
#     # QR-decomposition
#     # Q = np.linalg.qr(np.atleast_2d(data[:,pa]), 'complete')
#     # sigma = np.sum(Y**2) - np.sum((Y @ Q) ** 2)
#     # using solve
#     b = np.linalg.lstsq(X, Y, rcond=None)[0]
#     sigma = np.sum((Y - X @ b)**2)
#     likelihood = -0.5*n*( 1 + np.log(sigma/n))
#     return likelihood
