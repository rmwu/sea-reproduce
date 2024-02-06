import numpy as np
import ges
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
import itertools
from ges.scores.exp_gauss_int_l0_pen import ExpGaussIntL0Pen
np.set_printoptions(suppress=True)
k = 4
num_tot = 100
num_true = 0
for m in range(num_tot):
    while True:
        A = dag_avg_deg(k, 2)
        nxA = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
        nxA_undir = nx.DiGraph.to_undirected(nxA)
        if nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir):
            break
    W = A * np.random.uniform(1, 2, size=A.shape)
    scm = sempler.LGANM(W, (0, 15), (0, 0.2))
    n = 200000
    obs_data = scm.sample(n=n)
    obs_data2 = scm.sample(n=n)
    interv_data = scm.sample(n=n, do_interventions={1: (2, 10), 3: (2, 10)})
    interv_data2 = scm.sample(n=n, do_interventions={2: (2, 10)})
    data = [obs_data, interv_data, interv_data2]
    true_interv = [[], [1, 3], [2]]
    true_pdag = ges.exp_fit_bic(data, true_interv)[0]
    pdag_A = ges.utils.replace_unprotected(A, true_interv)
    if not np.all(pdag_A == true_pdag):
        num_tot -= 1
        continue
    A0 = np.zeros_like(A)

    max_interv = [[], []]
    max_pdag, max_score_change = ges.exp_fit_bic([obs_data, interv_data], max_interv)
    # max_score = ExpGaussIntL0Pen(data, max_interv).full_score(ges.utils.pdag_to_dag(max_pdag))

    for i in range(k):
        interv = [[], [i]]
        pdag, score_change = ges.exp_fit_bic([obs_data, interv_data], interv)
        # score = ExpGaussIntL0Pen(data, interv).full_score(ges.utils.pdag_to_dag(pdag))
        if score_change > max_score_change:
            max_interv = interv
            max_pdag = pdag
            max_score_change = score_change

    if set(max_interv[1]) <= set(true_interv[1]):
        num_true += 1
    print(max_interv)
print(num_true/num_tot)


