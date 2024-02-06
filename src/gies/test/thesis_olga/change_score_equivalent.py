# Copyright 2022 Olga Kolotuhina

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

import numpy as np
import ges
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
import itertools
from ges.scores.exp_gauss_int_l0_pen import ExpGaussIntL0Pen
import time

np.set_printoptions(suppress=True)


k = 11
num_interv = 2
list_interv = []
count_time = 0
for L in range(k + 1):
    for subset in itertools.combinations(range(k), L):
        list_interv.append(list(subset))

sets = []
for subset in itertools.product(list_interv, repeat=num_interv - 1):
    sets.append([[]] + list(subset))
try:
    num_iter = 10
    num_equal = 0
    iter = 0
    count = 4
    while iter < num_iter:
        tic = time.time()
        A = dag_avg_deg(k, 2)
        nxA = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
        nxA_undir = nx.DiGraph.to_undirected(nxA)
        if not (nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir)):
            continue
        # print(A)
        # A = np.array([[0, 1, 1, 0, 1, 1],
        #              [0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 0, 0, 0],
        #              [0, 1, 1, 0, 0, 1],
        #              [0, 1, 0, 1, 0, 0],
        #              [0, 0, 0, 0, 0, 0]])
        np.random.seed(count)
        print(count)
        count += 1
        W = A * np.random.uniform(1, 2, size=A.shape)
        scm = sempler.LGANM(W, (0, 15), (0, 0.2))
        n = 30000
        obs_data = scm.sample(n=n)
        obs_data2 = scm.sample(n=n)
        interv_data = scm.sample(n=n, do_interventions={1: (2, 10)})
        # interv_data2 = scm.sample(n=n, do_interventions={2: (2, 10)})
        data = [obs_data, interv_data]
        A0 = np.zeros_like(A)

        score_change = []
        pdags = []
        full_score_pdag = []
        full_score_empty = []

        for set in sets:
            P, P_score_change = ges.exp_fit_bic(data, set, debug=0)
            score_change.append(P_score_change)
            pdags.append(P)
            # score = ExpGaussIntL0Pen(data, set)
            # full_score_pdag.append(score.full_score(ges.utils.pdag_to_dag(P)))
            # full_score_empty.append(score.full_score(A0))
        # for (ind, i) in enumerate(score_change):
        #     print(sets[ind], i)
        # print("\n")
        # for i in full_score_pdag:
        #     print(i)
        # print("\n")

        # print(max(score_change))
        # max_elts = np.isclose(score_change, max(score_change), rtol=1e-10)
        # for i, set in enumerate(sets):
        # print(set, score_change[i])
        # max_change = [set for set, elt in zip(sets, max_elts) if elt == True]
        # print(max_change)
        # print(max(full_score_pdag))
        # max_elts = np.isclose(full_score_pdag, max(full_score_pdag))
        # print([set for set, elt in zip(sets, max_elts) if elt == True])

        # print([[], [1]] in max_change)
        # print(A)

        count_markov_eqv = 0
        markov_eqv = []
        true_P = ges.utils.dag_to_cpdag(A)
        A_copy = A.copy()
        for i in [1]:
            A_copy[:, i] = 0
        true_P_int = ges.utils.dag_to_cpdag(A_copy)
        for ind, P in enumerate(pdags):
            if np.all(ges.utils.dag_to_cpdag(ges.utils.pdag_to_dag(P)) == true_P):
                P_int = P.copy()
                for i in sets[ind]:
                    P_int[:, i] = 0
                if np.all(
                    ges.utils.dag_to_cpdag(ges.utils.pdag_to_dag(P_int)) == true_P_int
                ):
                    count_markov_eqv += 1
                    # print(sets[ind])
                    # print(pdags[ind])
        true_index = sets.index([[], [1]])
        elts = np.isclose(score_change, score_change[true_index], rtol=1e-10)
        change = [set for set, elt in zip(sets, elts) if elt]
        pdags_change = [p for p, elt in zip(pdags, elts) if elt]
        # for i, set in enumerate(change):
        #     print(set, "\n", pdags_change[i])
        #     print(ges.exp_fit_bic(data, set))
        #     print(ExpGaussIntL0Pen(data, set).sample_cov)
        #     print(ExpGaussIntL0Pen(data, set).part_sample_cov)
        # for i, set in enumerate(sets):
        #     print(set, "\n", pdags[i])
        #     print(ges.exp_fit_bic(data, set, debug=0))
        #     print(ExpGaussIntL0Pen(data, set).sample_cov)
        #     print(ExpGaussIntL0Pen(data, set).part_sample_cov)
        if np.all(ges.utils.replace_unprotected(A, [[], [1]]) == pdags[true_index]):
            iter += 1
        else:
            continue
        if len(change) == count_markov_eqv:
            num_equal += 1
            print(time.time() - tic)
            count_time += time.time() - tic
        else:
            print(A)
            break
        print(iter)

    print(f"All {num_iter} iterations successful: {num_equal == num_iter}")
    print(num_equal)

except KeyboardInterrupt:
    print(iter == num_equal)

print(count_time / num_iter)
