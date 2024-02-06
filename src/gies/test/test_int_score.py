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

import subprocess
import unittest
import numpy as np
import sempler, sempler.generators
import gies
import os
from gies.scores.gauss_int_l0_pen import GaussIntL0Pen


# Tests for the GaussIntL0Pen score and tests comparing with the PCALG implementation


class ScoreTests(unittest.TestCase):
    true_A = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    W = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(W, (0, 2), (0, 1))
    p = len(W)
    n = 10000
    n1 = 10000
    n2 = 10000
    obs_data = scm.sample(n=n)
    interv_data_1 = scm.sample(n=n1, do_interventions={1: (1, 3)})
    interv_data_2 = scm.sample(n=n2, do_interventions={2: (2, 3)})
    data = [obs_data, interv_data_1, interv_data_2]
    interv = [[], [1], [2]]
    # centering the data
    center_data = [data_x - np.mean(data_x, axis=0) for data_x in data]

    score = GaussIntL0Pen(data, interv)
    A_gies, score_gies_change = gies.fit_bic(data, interv)

    def test_fullscore_true_vs_empty(self):
        print("Score of true vs empty graph")
        # Compute score of the true DAG
        true_score = self.score.full_score(self.true_A)
        self.assertIsInstance(true_score, float)
        # Compute score of unconnected graph
        score_empty = self.score.full_score(np.zeros((self.p, self.p)))
        self.assertIsInstance(score_empty, float)
        print("True DAG vs empty:", true_score, score_empty)
        self.assertGreater(true_score, score_empty)

    def test_score_decomposability_obs(self):
        # As a black-box test, make sure the score functions
        # preserve decomposability
        print("Decomposability of observational score")
        # Compute score of the true DAG
        full_score = self.score.full_score(self.true_A)
        # Compute score of true DAG using local scores
        acc = 0
        for (j, pa) in self.factorization:
            local_score = self.score.local_score(j, pa)
            print("  ", j, pa, local_score)
            acc += local_score
        print("Full vs. acc:", full_score, acc)
        self.assertAlmostEqual(full_score, acc, places=2)

    def test_fullscore_gies_vs_true_score(self):
        print("The fullscore of the GIES PDAG vs the true score")
        # Compute a consistent extension of the PDAG and its score
        A_gies_dag = gies.utils.pdag_to_dag(self.A_gies)
        score_full_gies = self.score.full_score(A_gies_dag)
        self.assertIsInstance(score_full_gies, float)
        # Compute score of the true DAG
        true_score = self.score.full_score(self.true_A)
        self.assertIsInstance(true_score, float)
        print("GIES PDAG vs true DAG :", score_full_gies, true_score)
        self.assertAlmostEqual(score_full_gies, true_score, places=2)

    def test_fullscore_all_dags(self):
        print("Scores of all consistent extensions of the PDAG returned by GIES")
        cpdag_A = gies.utils.replace_unprotected(self.true_A)
        dags = gies.utils.pdag_to_all_dags(cpdag_A)
        score_dags = list(np.zeros(len(dags)))
        for index, dag in enumerate(dags):
            score_dags[index] = self.score.full_score(dag)
            if index > 1:
                self.assertAlmostEqual(
                    score_dags[index - 1], score_dags[index], places=2
                )
        print("all DAGS from PDAG", score_dags)
