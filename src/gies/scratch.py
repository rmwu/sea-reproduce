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

"""
Module with old/discarded functions.
"""


def exp_fit_bic(
    data,
    interv,
    mean=[],
    sigma=[],
    K=[],
    A0=None,
    phases=["forward", "backward", "turning"],
    iterate=True,
    debug=0,
):
    """Run GES on the given data, using the Gaussian BIC score
    (l0-penalized Gaussian Likelihood). The data is not assumed to be
    centered, i.e. an intercept is fitted.

    To use a custom score, see gies.fit.

    Parameters
    ----------
    data : list of numpy.ndarray
        every matrix in the list corresponds to an environment,
        the n x p matrix containing the observations, where columns
        correspond to variables and rows to observations.
    interv: a list of lists
        a list of the interventions sets which
        corresponds to the environments in data
    A0 : numpy.ndarray, optional
        The initial I-essential graph on which GIES will run, where where `A0[i,j]
        != 0` implies the edge `i -> j` and `A[i,j] != 0 & A[j,i] !=
        0` implies the edge `i - j`. Defaults to the empty graph
        (i.e. matrix of zeros).
    phases : [{'forward', 'backward', 'turning'}*], optional
        Which phases of the GIES procedure are run, and in which
        order. Defaults to `['forward', 'backward', 'turning']`.
    iterate : bool, default=False
        Indicates whether the given phases should be iterated more
        than once.
    debug : int, optional
        If larger than 0, debug are traces printed. Higher values
        correspond to increased verbosity.

    Returns
    -------
    estimate : numpy.ndarray
        The adjacency matrix of the estimated I-essential graph
    total_score : float
        The score of the estimate.

    Raises
    ------
    TypeError:
        If the type of some of the parameters was not expected,
        e.g. if data is not a numpy array.
    ValueError:
        If the value of some of the parameters is not appropriate,
        e.g. a wrong phase is specified.

    Example
    -------

    Data from a linear-gaussian SCM (generated using
    `sempler <https://github.com/juangamella/sempler>`__)

    >>> import numpy as np
    >>> data = [np.array([[3.23125779, 3.24950062, 13.430682, 24.67939513],
    ...                  [1.90913354, -0.06843781, 6.93957057, 16.10164608],
    ...                  [2.68547149, 1.88351553, 8.78711076, 17.18557716],
    ...                  [0.16850822, 1.48067393, 5.35871419, 11.82895779],
    ...                  [0.07355872, 1.06857039, 2.05006096, 3.07611922]])]
    >>> interv = [[]]

    Run GES using the gaussian BIC score:

    >>> import ges
    >>> gies.fit_bic(data, interv)
    (array([[0, 1, 1, 0],
           [0, 0, 0, 0],
           [1, 1, 0, 1],
           [0, 1, 1, 0]]), 15.674267611628233)
    """
    # Initialize Gaussian BIC score (precomputes scatter matrices, sets up cache)
    cache = ExpGaussIntL0Pen(data, interv, mean, sigma, K)
    # Unless indicated otherwise, initialize to the empty graph
    A0 = np.zeros((cache.p, cache.p)) if A0 is None else A0
    return fit(cache, A0, phases, iterate, debug)
