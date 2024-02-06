import numpy as np
import ges
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
import itertools
from ges.scores.exp_gauss_int_l0_pen import ExpGaussIntL0Pen
np.set_printoptions(suppress=True)
k = 7
while True:
    A = dag_avg_deg(k, 2)
    nxA = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
    nxA_undir = nx.DiGraph.to_undirected(nxA)
    if nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir):
        break
W = A * np.random.uniform(1, 2, size=A.shape)
scm = sempler.LGANM(W, (0, 15), (0, 0.2))
n = 20000
obs_data = scm.sample(n=n)
obs_data2 = scm.sample(n=n)
interv_data = scm.sample(n=n, do_interventions={1: (2, 10), 2: (3, 10)})
# interv_data2 = scm.sample(n=n, do_interventions={2: (2, 10)})
data = [obs_data, interv_data]
A0 = np.zeros_like(A)

score_change = []
pdags = []
full_score_pdag = []
full_score_empty = []
list_interv = []

max_interv = [[], []]
max_P = ges.exp_fit_bic(data, max_interv)[0]
max_score = ExpGaussIntL0Pen(data, max_interv).full_score(ges.utils.pdag_to_dag(max_P))

for i in range(k):
    interv = [[], [i]]
    P = ges.exp_fit_bic(data, interv)[0]
    score = ExpGaussIntL0Pen(data, interv).full_score(ges.utils.pdag_to_dag(P))
    if score > max_score:
        max_score = score
        max_P = P
        max_interv = interv

print(max_score, max_P, max_interv)
print(A)
