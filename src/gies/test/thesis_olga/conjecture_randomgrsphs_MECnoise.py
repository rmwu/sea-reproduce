import numpy as np
from ges import ExpGaussIntL0Pen
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
from sempler.generators import intervention_targets
from ges import utils
import itertools
import time


np.set_printoptions(suppress=True)

count = 0
count_time = 0
rep = 100
num_nodes = 4
num_interv = 1
deg = (num_nodes-1)/2
# w_min=(num_nodes-1)/2
for iter in range(rep):
    # print(iter)
    tic = time.time()
    while True:
        true_G = dag_avg_deg(num_nodes, deg)
        nxA = nx.from_numpy_array(true_G, create_using=nx.MultiDiGraph())
        nxA_undir = nx.DiGraph.to_undirected(nxA)
        if nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir):
            break
    print("True Graph:", true_G)
    true_interv = [[]] + intervention_targets(num_nodes, num_interv, (1, num_nodes))
    print("True intervention", true_interv)
    true_G_interv = []
    for interv in true_interv:
        true_G_interv.append(utils.intervened_graph(true_G, interv))

    W = true_G * np.random.uniform(1, 2, size=true_G.shape)
    scm = sempler.LGANM(W, (0, 15), (0, 0.2))
    n = 20000
    data = []
    for i in true_interv:
        interventions = {}
        for t in i:
            interventions[t] = (2, 10)
        data.append(scm.sample(n=n, do_interventions=interventions))

    G0 = np.zeros_like(true_G)
    empty_scores = []

    true_score_G0 = ExpGaussIntL0Pen(data, true_interv).full_score(G0)
    true_score_G = ExpGaussIntL0Pen(data, true_interv).full_score(true_G)
    true_score_change = true_score_G - true_score_G0

    list_targets = []
    for L in range(num_nodes+1):
        for subset in itertools.combinations(range(num_nodes), L):
            list_targets.append(list(subset))

    all_interv = []
    for subset in itertools.product(list_targets, repeat=num_interv):
        I = [[]] + list(subset)
        all_interv.append(I)
        empty_scores.append(ExpGaussIntL0Pen(data, I).full_score(G0))

    noisy = dag_avg_deg(num_nodes, 1) + true_G
    noisy_true_G  = utils.pdag_to_dag(np.where(noisy == 1, noisy, 0))
    print("Noisy True Graph", noisy_true_G)
    MEC_dags = utils.pdag_to_all_dags(utils.dag_to_cpdag(noisy_true_G))

    true_G_unprotected = utils.replace_unprotected(true_G, true_interv)
    noisy_true_G_unprotected = utils.replace_unprotected(noisy_true_G, true_interv)
    equiv = np.all(true_G_unprotected == noisy_true_G_unprotected)

    if equiv:
        continue
    for (ind1, I) in enumerate(all_interv):
        for G in MEC_dags:
            score_G_I = ExpGaussIntL0Pen(data, I).full_score(G)
            score_change_G_I = score_G_I - empty_scores[ind1]
            score_change_equal = np.isclose(score_change_G_I, true_score_change, rtol=1e-10)
            if score_change_equal:
                print(G)
                print(I)
                count += 1
            equivalence = True
            for (ind2, interv) in enumerate(I):
                G_interv = utils.intervened_graph(G, interv)
                if not utils.check_markov_equiv(G_interv, true_G_interv[ind2]):
                    equivalence = False
                    break
    count_time += time.time()-tic

print(count)
print(count_time/rep)

