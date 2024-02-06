# Copyright 2020 Juan Luis Gamella Martin

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
from . import log_likelihood
from . import log_likelihood_means
from .decomposable_score import DecomposableScore

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for samples from multiple
# environments where certain variables have received noise interventions


class ExperimentalScore(DecomposableScore):
    """
    Attributes
    ----------
    """

    def __init__(self, data, centered=True, fine_grained=True, lmbda=None, tol=1e-16, max_iter=10, debug=0):
        super().__init__(data, cache=False, debug=debug)
        self.e = len(data)
        self.p = data[0].shape[1]
        self.n_obs = np.array([len(env) for env in data])
        self.N = sum(self.n_obs)
        self.lmbda = 0.5 * np.log(self.N) if lmbda is None else lmbda
        self.max_iter = max_iter
        self.tol = tol
        self.fine_grained = fine_grained
        self.centered = centered

        # Precompute scatter matrices
        self._sample_covariances = np.array([np.cov(env, rowvar=False, ddof=0) for env in data])
        # Precompute sample means
        if not centered:
            self._sample_means = np.array([np.mean(env, axis=0) for env in data])

    def full_score(self, A, I):
        """
        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a collection of samples from different environments, by finding
        the maximum likelihood estimates of the corresponding connectivity
        matrix (weights) and noise term variances.
        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
        I : list of sets
            the sets of variables which have received interventions in
            each environment
        Returns
        -------
        score : float
            the penalized log-likelihood score
        """
        I = I if self.fine_grained else [I] * self.e
        # Compute MLE and its likelihood
        if self.centered:
            B, omegas = self._mle_full(A, I)
            likelihood = log_likelihood.full(B, omegas, self._sample_covariances, self.n_obs)
        else:
            B, nus, omegas = self._mle_full(A, I)
            likelihood = log_likelihood_means.full(B, nus, omegas, self._data)
        #   Note: the number of parameters is the number of edges +
        #   the total number of marginal variances/means, which depends on
        #   the number of interventions.
        l0_term = self.lmbda * ddof_full(A, I, centered=self.centered)
        score = likelihood - l0_term
        return score

    def local_score(self, x, pa, I):
        """Return the local score of a given node and a set of
        parents. If self.cache=True, will use previously computed
        score if possible.
        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents
        I : list of sets
            the sets of variables which have received interventions in
            each environment
        Returns
        -------
        score : float
            the corresponding score
        """
        I = I if self.fine_grained else [I] * self.e
        return self._compute_local_score(x, pa, I)
        # if self._cache is None:
        #     return self._compute_local_score(x, pa, I)
        # else:
        #     key = (x, tuple(sorted(pa)))
        #     try:
        #         score = self._cache[key]
        #         print("score%s: using cached value %0.2f" % (key,score)) if self._debug>=2 else None
        #     except KeyError:
        #         score = self._compute_local_score(x, pa, I)
        #         self._cache[key] = score
        #         print("score%s = %0.2f" % (key,score)) if self._debug>=2 else None
        #     return score

    def _compute_local_score(self, x, pa, I):
        """
        Given a node and its parents, return the local l0-penalized
        log-likelihood of a collection of samples from different
        environments, by finding the maximum likelihood estimates of the
        weights and noise term variances.
        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents
        I : list of sets
            the sets of variables which have received interventions in
            each environment
        Returns
        -------
        score : float
            the penalized log-likelihood score
        """
        # Compute MLE, with the local subgraph p -> x for p in pa
        #   Only do set -> list conversion once, as the ordering is not
        #   guaranteed to be consistent.
        if self.centered:
            b, omegas = self._mle_local(x, pa, I)
            likelihood = log_likelihood.local(x, b, omegas, self._sample_covariances, self.n_obs)
        else:
            b, nus, omegas = self._mle_local(x, pa, I)
            likelihood = log_likelihood_means.local(x, b, nus, omegas, self._data)
        #  Note: the number of parameters is the number of parents (one
        #  weight for each) + one marginal variance/mean per environment for x
        l0_term = self.lmbda * ddof_local(x, pa, I, centered=self.centered)
        score = likelihood - l0_term
        return score

    # --------------------------------------------------------------------
    #  Functions for the maximum likelihood estimation of the
    #  weights/variances
    def _mle_full(self, A, I):
        """Given the DAG adjacency A and observations, compute the maximum
        likelihood estimate of the connectivity weights and noise
        variances, returning them and the resulting log likelihood.
        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
        Returns
        -------
        B : np.array
            the MLE estimate of the connectivity matrix (the weights)
        omegas : np.array
            the MLE estimate of the noise variances of each variable
        """
        B, omegas = _alternating_mle(A, I, self._sample_covariances,
                                     self.n_obs, self.tol, self.max_iter, self._debug)
        if self.centered:
            return B, omegas
        else:
            noise_term_means = _noise_means_from_B(B, I, self._sample_means, self.n_obs)
            return B, noise_term_means, omegas

    def _mle_local(self, j, pa, I):
        pa = list(pa)
        A = np.zeros((1 + len(pa), 1 + len(pa)))
        A[1:, 0] = 1
        sub_I = [set(range(1, len(pa) + 1)) for _ in I]
        for k, targets in enumerate(I):
            if j in targets:
                # print(k, j, targets)
                sub_I[k].add(0)
        # print(I)
        # print(sub_I)
        sub_covariances = np.array([sigma[[j] + pa, :][:, [j] + pa]
                                    for sigma in self._sample_covariances])
        B, Omegas = _alternating_mle(A, sub_I, sub_covariances, self.n_obs,
                                     self.tol, self.max_iter, self._debug)
        b = np.zeros(self.p)
        b[pa] = B[1:, 0]
        omegas = Omegas[:, 0]  # Extract marginal variances of x
        if self.centered:
            return b, omegas
        else:
            sub_means = self._sample_means[:, [j] + pa]
            nus = _noise_means_from_B(B, sub_I, sub_means, self.n_obs)[:, 0]
            return b, nus, omegas

# --------------------------------------------------------------------
# Wrapper class


class FixedInterventionalScore(ExperimentalScore):

    def __init__(self, data, I, centered=True, fine_grained=False, lmbda=None, tol=1e-16, max_iter=10, debug=0):
        self.I = I
        super().__init__(data, centered, fine_grained, lmbda, tol, max_iter, debug)

    def full_score(self, A):
        return super().full_score(A, self.I)

    def local_score(self, x, pa):
        return super().local_score(x, pa, self.I)


# --------------------------------------------------------------------
# Functions for the alternating maximization procedure used to find
# the MLE

def _alternating_mle(A, I, sample_covariances, n_obs, tol=1e-16, max_iter=50, debug=0):
    """Given the DAG adjacency A and observations, compute the maximum
    likelihood estimate of the connectivity weights and noise
    variances using an alternating maximization procedure.
    Parameters
    ----------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
    sample_covariances : list(np.array())
        the list of sample covariances (see np.cov) of the sample from
        each environment
    n_obs : list of ints
       the number of observations available from each environment
       (i.e. the sample size)
    tol : float
        The tolerance for the alternating maximization procedure (see
        em_like below) used to find the MLE, i.e. the procedure will
        stop when the maximum L1 distance between succesive parameters
        (weights and variances) is less than tol.
    max_iter : int
        The maximum number of iterations for which the alternating
        maximization procedure is run (see em_like below).
    debug : bool
        if debug traces should be printed
    Returns
    -------
    B : np.array
        the MLE estimate of the connectivity matrix (the weights)
    omegas : np.array
        the MLE estimate of the noise variances of each variable
    """
    # Set starting point for optimization procedure
    p = len(A)
    B_0 = np.zeros((p, p))
    avg_covariance = sample_covariances.mean(axis=0)
    for j in range(p):
        pa = list(np.where(A[:, j] != 0)[0])
        B_0[pa, j] = _regress(j, pa, avg_covariance)
    # Set up alternating minimization procedure:
    omegas_0 = _omegas_from_B(B_0, [set()] * len(I), sample_covariances, n_obs)  # e_func(B_0)

    # 1. "Expectation" function: given B compute variances of noise terms

    def e_func(B):
        return _omegas_from_B(B, I, sample_covariances, n_obs)

    # 2. "Maximization" function: given variances of noise terms, compute B

    def m_func(omegas):
        return _B_from_omegas(omegas, A, sample_covariances, n_obs)
    # 3. Distance to check for convergence: max of L1 distance between B's and between omegas

    def dist(prev_B, prev_omegas, B, omegas):
        return max(abs(B - prev_B).max(), abs(omegas - prev_omegas).max())

    # 4. Keep track of the objective function if debugging is desired

    def objective(B, omegas):
        return log_likelihood.full(B, omegas, sample_covariances, n_obs) if debug else None

    # Run procedure
    print("    Running alternating optimization procedure") if debug else None
    (B, omegas) = _em_like(e_func, m_func, B_0, omegas_0, dist,
                           tol=tol, max_iter=max_iter, objective=objective, debug=debug)
    return B, omegas


def _regress(j, pa, cov):
    # compute the regression coefficients from the
    # empirical covariance (scatter) matrix i.e. b =
    # Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
    return np.linalg.solve(cov[pa, :][:, pa], cov[j, pa])


def _B_from_omegas(omegas, A, sample_covariances, n_obs):
    """
    Part of the alternating likelihood maximization procedure. Given
    fixed noise variances and a DAG adjacency A, return the maximizing
    connectivity matrix.
    Parameters
    ----------
    omegas : np.array
        the noise variances of each variable
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
    sample_covariances : list(np.array())
        the list of sample covariances (see np.cov) of the sample from
        each environment
    n_obs : list of ints
       the number of observations available from each environment
       (i.e. the sample size)
    Returns
    -------
    B : np.array
        the maximizing connectivity matrix
    """
    e = len(sample_covariances)
    p = len(sample_covariances[0])
    B = np.zeros((p, p))
    for j in range(p):
        # Dividing by j's variances seems a bit weird at first, but
        # note that the regression below is done only for j
        weights = (n_obs / omegas[:, j])
        weights /= sum(weights)
        pooled_covariance = np.sum(sample_covariances * np.reshape(weights, (e, 1, 1)), axis=0)
        parents = np.where(A[:, j] != 0)[0]
        coef = _regress(j, parents, pooled_covariance)
        B[parents, j] = coef
    return B


def _noise_means_from_B(B, I, sample_means, n_obs):
    e, p = sample_means.shape
    I_B = (np.eye(p) - B)
    # This gives the noise term means for each environment (as if all
    # variables were intervened, on all environments)
    means = sample_means @ I_B
    # Compute the noise means for each variable across the
    # non-intervened environments
    scaled = sample_means * np.reshape(n_obs, (e, 1))
    for j in range(p):
        non_intervened = np.where([j not in i for i in I])[0]
        if len(non_intervened) > 0:
            # pooled is a 1xp array, containing the mean for each
            # variable, pooled across the "non_intervened" environments
            pooled = scaled[non_intervened].sum(axis=0) / n_obs[non_intervened].sum()
            means[non_intervened, j] = pooled @ I_B[:, j]
    return means


def _omegas_from_B(B, I, sample_covariances, n_obs):
    """
    Part of the alternating likelihood maximization procedure. Given
    fixed noise variances and a DAG adjacency A, return the maximizing
    connectivity matrix.
    Parameters
    ----------
    omegas : np.array
        the noise variances of each variable
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j
    sample_covariances : list(np.array())
        the list of sample covariances (see np.cov) of the sample from
        each environment
    n_obs : list of ints
       the number of observations available from each environment
       (i.e. the sample size)
    Returns
    -------
    B : np.array
        the maximizing connectivity matrix
    """
    e = len(sample_covariances)
    if len(I) != e:
        raise ValueError(
            "Wrong number of intervention targets (%d) for %d environments" % (len(I), e))
    p = len(B)
    omegas = np.zeros((e, p))
    # Scale sample covariances wrt. each environment's sample size
    scaled = sample_covariances * np.reshape(n_obs, (e, 1, 1))
    for j in range(p):
        # parents = np.where(B[:, j] != 0)[0]
        # Separate into environments where j is intervened/not
        intervened = np.where([j in i for i in I])[0]
        non_intervened = np.where([j not in i for i in I])[0]
        # Compute variance for environments where variable was not
        # intervened (variance is fixed across them)
        if len(non_intervened) > 0:
            pooled = scaled[non_intervened].sum(axis=0) / sum(n_obs[non_intervened])
            # variance = pooled[j,j] - pooled[j,parents] @ B[parents,j]
            variance = ((np.eye(p) - B).T @ pooled @ (np.eye(p) - B))[j, j]
            omegas[non_intervened, j] = variance
        # Compute variance for other environments
        for k in intervened:
            # TODO: Fix this computational waste (computing a matrix product to extract one element)
            variance = ((np.eye(p) - B).T @ sample_covariances[k] @ (np.eye(p) - B))[j, j]
            # variance = sample_covariances[k,j,j] - sample_covariances[k,j,parents] @ B[parents,j]
            # Why does the above (commented) not work?  Idea: We're
            # not asking about the variance of the conditional
            # distribution of Xj given its parents in environment e,
            # as this would mean the regression coefficients would be
            # allowed to change within environments. We want to know
            # the variance of the residuals resulting from using the
            # regression coefficients in B.
            omegas[k, j] = variance
    return omegas  # abs(omegas)


def _em_like(e_func, m_func, M_0, E_0, dist, tol=1e-6, assume_convex=False, max_iter=100, objective=None, debug=False):
    """
    EM-like alternating optimization procedure.
    Parameters
    ----------
    e_func : function
        "Expectation" function
    m_func : function
        "Maximization" function
    M_0 : any
        initial value for the expectation function
    E_0 : any
        initial value for the maximization function. In principle, not
        needed as the first step of the algorithm overrides this
        value; however, if the initial values satisfy convergence this
        allows us to return them : in a user-transparent way).
    dist : function
        The distance function used to check convergence. Takes as
        arguments the E,M from the previous and current iterations and
        returns a float
    tol : float
        Convergence tolerance. Stop if distance between consecutive
        iterations is below this threshold.
    assume_convex : boolean
        if the objective is assumed convex. If true, an increase in
        the distance of iterations is due to numerical issues, and we
        stop the iterations at this point, even if the tolerance has
        not been reached
    objective : function
        For debugging purposes, the objective function
    debug : boolean
        Print results of each iteration
    Returns
    -------
    M : any
        the approximated maximizer of the "Maximization quantity"
    E : any
        the approximated maximizer of the "Expectation quantity"
    """
    print("     ", end="") if debug else None
    prev_E = E = E_0
    prev_M = M = M_0
    prev_delta = np.inf
    for i in range(max_iter):
        E = e_func(M)  # E-step
        M = m_func(E)  # M-step
        delta = dist(prev_M, prev_E, M, E)
        # Debug outputs
        if objective is not None and debug:
            try:
                value = objective(M, E)
                print(" %0.16f (%0.16f)" % (delta, value), end="")
            except Exception as e:
                print(" %0.16f (%s)" % (delta, e), end="")
        elif debug:
            print(" %0.4f" % delta, end="")
            # Check stopping conditions
        if delta <= tol:
            print() if debug else None
            return M, E
        elif assume_convex and delta > prev_delta:
            # if program is convex, an increase in the distance
            # between iterations is due to numerical issues. Thus we
            # return the results from the previous iteration as we
            # assume that's as close as we can get
            print() if debug else None
            return prev_M, prev_E
        else:
            prev_E = E
            prev_M = M
            prev_delta = delta
    print(" MAX ITER REACHED") if debug else None
    return M, E


def ddof_full(A, I, centered=True):
    """Compute the number of free parameters in a model specified by the
    DAG adjacency and the intervention targets.
    Parameters
    ----------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.
    I : list of sets
        The sets of variables which have received interventions in
        each environment.
    centered : bool
        If we are also fitting an intercept or we assume the data to
        be centered.
    Returns
    -------
    ddof : int
        the number of free parameters in the model
    """
    p = len(A)
    # We estimate a weight for each edge
    no_edges = (A != 0).sum()
    # For each variable, we have to estimate its noise term variance
    # for each environment in which it receives an intervention, and
    # one for the rest (if any)
    no_variances = sum(len(i) for i in I) + p - len(set.intersection(*I))
    # Same goes for the noise term mean
    no_intercepts = 0 if centered else no_variances
    return no_edges + no_variances + no_intercepts


def ddof_local(j, pa, I, centered=True):
    """Compute the number of free parameters that are estimated for a
    variable in the model, given its parents and the intervention
    targets.
    Parameters
    ----------
    j : int
        The variable's index.
    pa : list(int)
        The set of parents.
    I : list of sets
        The sets of variables which have received interventions in
        each environment.
    centered : bool
        If we are also fitting an intercept or we assume the data to
        be centered.
    Returns
    -------
    ddof : int
        the number of free parameters
    """
    # We have to estimate the variance of the noise term for each
    # environment in which j receives an intervention, and one for the
    # rest (if any)
    no_variances = sum(j in i for i in I) + min(1, sum(j not in i for i in I))
    # Same goes for the noise term mean
    no_intercepts = 0 if centered else no_variances
    # We estimate one weight per parent
    return len(pa) + no_variances + no_intercepts
