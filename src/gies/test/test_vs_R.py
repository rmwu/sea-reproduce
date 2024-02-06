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


class vsRTests(unittest.TestCase):
    NUM_GRAPHS = 100
    R_DATA_DIR = "/tmp/gies_R_test/"
    os.makedirs(R_DATA_DIR, exist_ok=True)

    def test_vs_R(self):
        G = self.NUM_GRAPHS
        p = 10
        n = 1000
        rng = np.random.default_rng(42)
        for i in range(G):
            print("\n\nTesting case", i)
            W = sempler.generators.dag_avg_deg(p, 2.1, 0.5, 1, random_state=i)
            true_A = (W != 0).astype(int)
            # I = sempler.generators.intervention_targets(
            #     p, 4, 1, replace=False, random_state=i
            # )
            I = [[], [1], [2]]
            scm = sempler.LGANM(W, (0, 0), (1, 2), random_state=i)
            data = []
            for targets in I:
                interventions = dict((j, (0, rng.uniform(3, 4))) for j in targets)
                sample = scm.sample(
                    n=n, do_interventions=interventions, random_state=42
                )
                sample -= sample.mean(axis=0)
                data.append(sample)

            score = GaussIntL0Pen(data, I)

            # Tests vs. the PCALG R implementation
            # Save data for R, note that data must be centered
            datacsv = np.concatenate(data)
            np.savetxt(self.R_DATA_DIR + "data.csv", datacsv, delimiter=",")
            # Run GIES
            os.system("Rscript gies/test/run_gies.R")

            # Test score vs. what is computed by the R package PCALG
            gies_estimate, _ = gies.fit_bic(
                data, I, phases=["forward", "backward", "turning"], iterate=True
            )

            # Comparing the actual estimates
            print("Test estimate vs. PCALG estimate")
            A_pcalg = np.genfromtxt(self.R_DATA_DIR + "A_gies.csv")
            print("Python estimate")
            print(gies_estimate)
            print("PCALG estimate")
            print(A_pcalg)
            if not (gies_estimate == A_pcalg).all():
                print("Graph %d failed" % i)
                print(gies_estimate - A_pcalg)
            self.assertTrue((gies_estimate == A_pcalg).all())

            # Comparing empty and GIES estimate scores
            print("Test score vs. PCALG score")
            with open(self.R_DATA_DIR + "scores.csv") as f:
                pcalg_score_empty_graph = float(f.readline())
                pcalg_score_gies = float(f.readline())
            score_empty = score.full_score(np.zeros((p, p)))
            A_gies_dag = gies.utils.pdag_to_dag(gies_estimate)
            score_gies = score.full_score(A_gies_dag)
            self.assertAlmostEqual(pcalg_score_empty_graph, score_empty, places=5)
            print("empty score: python vs PCALG", score_empty, pcalg_score_empty_graph)
            self.assertAlmostEqual(pcalg_score_gies, score_gies, places=5)
            print("GIES estimate score: python vs pcalg", score_gies, pcalg_score_gies)
