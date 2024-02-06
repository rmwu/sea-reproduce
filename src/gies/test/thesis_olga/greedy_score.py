import numpy as np
import ges
import networkx as nx
import sempler
from sempler.generators import dag_avg_deg
from sempler.generators import  intervention_targets
import itertools
import random
import copy
from ges.scores.exp_gauss_int_l0_pen import ExpGaussIntL0Pen
from ges.scores.infinite_score import InfiniteScore
np.set_printoptions(suppress=True)


def forward(data, I_old, p, mean, sigma, K):
    I_max = copy.deepcopy(I_old)
    S_max = -np.inf
    max_dag_forw = np.zeros((p, p))
    for (index, I_index) in enumerate(I_old):
        if index == 0:
            continue
        list_check = list(set(range(p)) - set(I_index))
        for k in list_check:
            I = copy.deepcopy(I_old)
            I[index].append(k)
            pdag = ges.exp_fit_bic(data, I, mean, sigma, K)[0]
            dag = ges.utils.pdag_to_dag(pdag)
            full_score = ExpGaussIntL0Pen(data, I, mean, sigma, K).full_score(dag)
            if full_score > S_max:
                I_max = copy.deepcopy(I)
                S_max = full_score
                max_dag_forw = dag
    return I_max, S_max, max_dag_forw


def backward(data, I_old, ind_I, mean, sigma, K):
    I_max = copy.deepcopy(I_old)
    S_max = -np.inf
    max_dag_back = np.zeros((p, p))
    for (index, I_index) in enumerate(I_old):
        if index == 0:
            continue
        for i_back in I_index:
            I = copy.deepcopy(I_old)
            I[index].remove(i_back)
            pdag = ges.exp_fit_bic(data, I, mean, sigma, K)[0]
            dag = ges.utils.pdag_to_dag(pdag)
            full_score = ExpGaussIntL0Pen(data, I, mean, sigma, K).full_score(dag)
            if full_score > S_max:
                I_max = I
                S_max = full_score
                max_dag_back = dag
    return I_max, S_max, max_dag_back


num_tot = 1000
num_interv_targets = 2
for p in range(3, 10):
    list_interv = []
    for L in range(p+1):
        for subset in itertools.combinations(range(p), L):
            list_interv.append(list(subset))
        sets = []
        for subset in itertools.product(list_interv, repeat=num_interv_targets - 1):
            sets.append([[]] + list(subset))
    deg = random.uniform(p/2, p-1)
    print("# of nodes: ", p)
    max_full_score_true = 0
    max_inf_score_true = 0
    max_score_change_true = 0
    max_full_interv_True = 0
    max_change_interv_True = 0
    change_score_o_equiv = 0
    o_equiv_change_score = 0
    o_equiv = 0
    count = 0
    count_forw = 0
    count_greedy_o_equiv = 0
    count_greedy_o_equiv_forw = 0
    count_inf = 0
    count_inf_greedy = 0
    inf_count = 0
    inf_count_greedy = 0
    for m in range(num_tot):
        while True:
            A = dag_avg_deg(p, deg)
            nxA = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
            nxA_undir = nx.DiGraph.to_undirected(nxA)
            if nx.is_directed_acyclic_graph(nxA) and nx.is_connected(nxA_undir):
                W = A * np.random.uniform(0.5, 1, size=A.shape)
                scm = sempler.LGANM(W, (0, 0), (1, 2))
                n = 10
                true_interv = [[]] + intervention_targets(p, num_interv_targets - 1, (1, p))
                data = []
                for i in true_interv:
                    interventions = {}
                    for t in i:
                        interventions[t] = (0, 1)
                    data.append(scm.sample(n=n, do_interventions=interventions))
                mean = []
                sigma = []
                K = []
                for j in range(len(true_interv)):
                    mean.append(np.mean(data[j], axis=0))
                    sigma_temp = (np.diag(np.var(data[j], axis=0)))
                    sigma.append(sigma_temp)
                    K.append(np.linalg.inv(sigma_temp))
                [true_pdag, true_score_change] = ges.exp_fit_bic(data, true_interv)
                pdag_A = ges.utils.replace_unprotected(A, true_interv)
                if np.all(pdag_A == true_pdag):
                    break
        true_full_score = ExpGaussIntL0Pen(data, true_interv, mean, sigma, K).full_score(A)

        cov = []
        exp_val = []
        var = []
        precis = []
        for (num, i) in enumerate(true_interv):
            interventions = {}
            for t in i:
                interventions[t] = (0, 1)
            cov.append(scm.sample(population=True, do_interventions=interventions).covariance)
            exp_val.append(scm.sample(population=True, do_interventions=interventions).mean)
            var.append(np.diag(cov[num].diagonal()))
            precis.append(np.linalg.inv(var[num]))
        true_inf = InfiniteScore(cov, true_interv, exp_val, cov, precis).full_score(A)
        G_0 = np.zeros_like(A)
        max_interv = true_interv
        max_pdag = true_pdag
        max_score = true_full_score
        # max_inf_score = true_inf_score
        max_score_change = true_score_change
        max_change_interv = true_interv
        max_change_pdag = true_pdag
        for interv in sets:
            pdag, score_change = ges.exp_fit_bic(data, interv, mean, sigma, K)
            # print(pdag)
            dag = ges.utils.pdag_to_dag(pdag)
            full_score = ExpGaussIntL0Pen(data, interv, mean, sigma, K).full_score(dag)
            if np.isclose(score_change, true_score_change, rtol=1e-10):
                if ges.utils.check_o_equiv(dag, interv, A, true_interv):
                    inf = InfiniteScore(cov, interv, exp_val, cov, precis).full_score(dag)
                    change_score_o_equiv += 1
            if ges.utils.check_o_equiv(dag, interv, A, true_interv):
                if np.isclose(score_change, true_score_change, rtol=1e-10):
                    o_equiv_change_score += 1
            if full_score > max_score and not np.isclose(full_score, max_score, rtol=1e-10):
                max_interv = interv
                max_pdag = pdag
                max_score = full_score
            if score_change > max_score_change and not np.isclose(score_change, max_score_change, rtol=1e-10):
                max_score_change = score_change
                max_change_pdag = pdag
                max_change_interv = interv
        if max_interv == true_interv:
            max_full_interv_True += 1
        else:
            max_dag = ges.utils.pdag_to_dag(max_pdag)
            o_equiv += ges.utils.check_o_equiv(A, true_interv, max_dag, max_interv)
        max_dag = ges.utils.pdag_to_dag(max_pdag)
        if max_change_interv == true_interv:
            max_change_interv_True += 1
        if np.isclose(max_score, true_full_score, rtol=1e-10):
            max_full_score_true += 1
        if np.isclose(max_score_change, true_score_change, rtol=1e-10):
            max_score_change_true += 1

        max_interv_greedy = [[] for x in range(num_interv_targets)]
        pdag = ges.exp_fit_bic(data, max_interv_greedy, mean, sigma, K)[0]
        max_dag_greedy = pdag
        dag = ges.utils.pdag_to_dag(pdag)
        full_score = ExpGaussIntL0Pen(data, max_interv_greedy, mean, sigma, K).full_score(dag)
        max_score_greedy = full_score
        while True:
            max_score_old = max_score_greedy
            max_interv_forw_step, max_score_forw_step, max_dag_forw_step = forward(data, max_interv_greedy, p, mean, sigma, K)
            max_interv_back_step, max_score_back_step, max_dag_back_step = backward(data, max_interv_forw_step, p, mean, sigma, K)
            if max_score_old < max_score_back_step:
                max_interv_greedy = copy.deepcopy(max_interv_back_step)
                max_score_greedy = max_score_back_step
                max_dag_greedy = max_dag_back_step
            else:
                if max_score_forw_step > max_score_greedy:
                    max_interv_greedy = copy.deepcopy(max_interv_forw_step)
                    max_score_greedy = max_score_forw_step
                    max_dag_greedy = max_dag_forw_step
            if max_score_greedy == max_score_old:
                break

        max_interv_greedy_forw = [[] for x in range(num_interv_targets)]
        pdag = ges.exp_fit_bic(data, max_interv_greedy, mean, sigma, K)[0]
        dag = ges.utils.pdag_to_dag(pdag)
        max_dag_greedy_forw = dag
        full_score = ExpGaussIntL0Pen(data, max_interv_greedy_forw, mean, sigma, K).full_score(dag)
        max_score_greedy_forw = full_score
        while True:
            max_score_old_forw = max_score_greedy_forw
            max_interv_forw_step, max_score_forw_step, max_dag_forw_step = forward(data, max_interv_greedy_forw, p, mean, sigma, K)
            if max_score_forw_step > max_score_greedy_forw:
                max_interv_greedy_forw = copy.deepcopy(max_interv_forw_step)
                max_score_greedy_forw = max_score_forw_step
                max_dag_greedy_forw = max_dag_forw_step
            if max_score_greedy_forw == max_score_old_forw:
                break

        max_inf = InfiniteScore(cov, max_interv, exp_val, cov, precis).full_score(max_dag)
        if not max_interv == true_interv:
            inf_count += np.isclose(max_inf, true_inf, rtol=1e-10)
        if max_inf > true_inf and not np.isclose(max_inf, true_inf, rtol=1e-10):
            count_inf += 1

        max_inf_greedy = InfiniteScore(cov, max_interv_greedy, exp_val, cov, precis).full_score(max_dag_greedy)
        if not max_interv_greedy == true_interv:
            inf_count_greedy += np.isclose(max_inf_greedy, true_inf, rtol=1e-10)
        if max_inf_greedy > true_inf and not np.isclose(max_inf_greedy, true_inf, rtol=1e-10):
            count_inf_greedy += 1

        count_true = True
        for (ind, inter) in enumerate(max_interv):
            if set(max_interv[ind]) != set(max_interv_greedy[ind]):
                count_true = False
                if ges.utils.check_o_equiv(A, true_interv, max_dag_greedy, max_interv_greedy):
                    count_greedy_o_equiv += 1
                break
        if count_true:
            count += 1

        count_true = True
        for (ind, inter) in enumerate(max_interv):
            if set(max_interv[ind]) != set(max_interv_greedy_forw[ind]):
                count_true = False
                if ges.utils.check_o_equiv(A, true_interv, max_dag_greedy_forw, max_interv_greedy_forw):
                    count_greedy_o_equiv_forw += 1
                break
        if count_true:
            count_forw += 1
        print(m)

    print("Max Full Score equals true score", max_full_score_true)
    print("Max Interv is equal true intervention", max_full_interv_True)
    print("O_equivalent", o_equiv)
    print("Same score change, o-equivaleant", change_score_o_equiv)
    print("O-equivalent, same score change", o_equiv_change_score)
    print("Greedy algo correct", count)
    print("Greedy algo O-equiv", count_greedy_o_equiv)
    print("Greedy Forward algo correct", count_forw)
    print("Greedy Forward algo O-equiv", count_greedy_o_equiv_forw)
    print("Infinite score not true", count_inf)
    print("True Inf Score = Max Inf Score", inf_count)
    print("True Inf Score = Greedy Inf Score", inf_count_greedy)


