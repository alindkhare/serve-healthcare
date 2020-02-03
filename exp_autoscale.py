import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm
from datetime import datetime

from util import my_eval, get_accuracy_profile, get_latency_profile, read_cache, get_description

##################################################################################################
# tools
##################################################################################################
def random_sample(n_model, B, n_samples):
    """
    Input:
        n_model: number of models, n
        n_samples: 

    Output:
        B \in \{0,1\}^{n_samples \times n_model}
    """
    max_n_model = 10
    out = []
    i = 0
    while i < n_samples:
        # get a random probability of 1s and 0s
        pp = (max_n_model/n_model)*np.random.rand()
        # get random binary vector
        tmp = np.random.choice([0, 1], size=n_model, p=(1-pp,pp))
        # dedup
        for b in B:
            if np.array_equal(tmp, b):
                break
        out.append(tmp)
        i += 1
    return out

def nearby_sample(n_model, B_top, B, n_samples, dist=3):
    """
    """
    out = []
    i = 0
    while i < n_samples:
        # get a random b from B_top
        tmp = B_top[np.random.choice(list(range(len(B_top))))]
        # random binary vector near dist, by random filp 3 digits
        for j in range(dist):
            idx = np.random.choice(list(range(n_model)))
            if tmp[idx] == 0:
                tmp[idx] = 1
            else:
                tmp[idx] = 0
        # dedup
        for b in B:
            if np.array_equal(tmp, b):
                break
        out.append(tmp)
        i += 1
    return out

def get_obj(accuracy, latency, lamda, L, soft=True):
    if soft:
        return accuracy + lamda * (L - latency)
    else:
        if latency > L:
            return 0.0
        else:
            return accuracy

def write_res(V, c, b, method):
    """
    profile and write a line
    """
    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std = get_accuracy_profile(V, b, return_all=True)
    tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
    latency = np.percentile(tmp_latency, 95)

    with open(log_name, 'a') as fout:
        fout.write('{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}\n'.format(c[0], c[1], method, roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std, latency, b))

def write_traj(V, c, b, method):
    """
    more detailed results than write_res
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std = get_accuracy_profile(V, b, return_all=True)
    tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
    latency = np.percentile(tmp_latency, 95)

    with open(traj_name, 'a') as fout:
        fout.write('{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}\n'.format(current_time, c[0], c[1], method, roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std, latency, b))

##################################################################################################
# naive
##################################################################################################
def solve_random(V, c, L, lamda):
    """
    random incremental
    """
    print("="*60)
    print("start solve_random")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_all = list(range(n_model))
    latency = 0.0
    for i in range(n_model):
        tmp_idx = np.random.choice(idx_all)
        idx_all.remove(tmp_idx)
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        latency = np.percentile(tmp_latency, 95)
        print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_random')

    opt_b = b
    write_res(V, c, opt_b, 'solve_random')

    return opt_b

def solve_greedy_accuracy(V, c, L, lamda):
    """
    greedy accuracy incremental
    """
    print("="*60)
    print("start solve_greedy_accuracy")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 2])[::-1] # the 3rd col is accuracy
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        latency = np.percentile(tmp_latency, 95)
        print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_greedy_accuracy')
    print('found best b is: {}'.format(b))

    opt_b = b
    write_res(V, c, opt_b, 'solve_greedy_accuracy')

    return opt_b

def solve_greedy_latency(V, c, L, lamda):
    """
    greedy latency incremental
    """
    print("="*60)
    print("start solve_greedy_latency")
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 3]) # the 4th col is latency
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        latency = np.percentile(tmp_latency, 95)
        print('model id: ', tmp_idx, 'b: ', b, latency)
        if latency >= L:
            b[tmp_idx] = 0
            break
        write_traj(V, c, b, 'solve_greedy_latency')
    print('found best b is: {}'.format(b))

    opt_b = b
    write_res(V, c, opt_b, 'solve_greedy_latency')

    return opt_b

##################################################################################################
# opt
##################################################################################################
def solve_opt_passive(V, c, L, lamda):

    global opt_b_solve_random
    global opt_b_solve_greedy_accuracy
    global opt_b_solve_greedy_latency

    print("="*60)
    print("start solve_opt_passive")
    # --------------------- hyper parameters ---------------------
    
    if global_debug:
        N1 = 1
    else:
        N1 = 100 # warm start

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    try:
        B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
        print('warm init success', B)
    except:
        print('warm init failed')
        if global_debug:
            B = []
        else:
            opt_b_solve_random = solve_random(V, c, L, lamda)
            opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
            opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)
            B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
    Y_accuracy = []
    all_latency = []
    Y_latency = []

    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}

    # --------------------- (1) warm start ---------------------
    B = B + random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        print('latency: ', Y_latency[-1])

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
    print(all_obj)
    opt_idx = np.argmax(np.nan_to_num(all_obj))
    opt_b = B[opt_idx]
    write_traj(V, c, opt_b, 'solve_opt_passive')
    write_res(V, c, opt_b, 'solve_opt_passive')

    return opt_b

def solve_opt_active(V, c, L, lamda):

    global opt_b_solve_random
    global opt_b_solve_greedy_accuracy
    global opt_b_solve_greedy_latency

    print("="*60)
    print("start solve_opt_active")
    # --------------------- hyper parameters ---------------------
    
    if global_debug:
        N1 = 1 # warm start
        topK = 1 # search near top
        N3 = 1 # profile
        epoches = 1
    else:
        N1 = 100 # warm start
        topK = 3 # search near top
        N3 = 30 # profile
        epoches = 10

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    try:
        B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
        print('warm init success', B)
    except:
        print('warm init failed')
        if global_debug:
            B = []
        else:
            opt_b_solve_random = solve_random(V, c, L, lamda)
            opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
            opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)
            B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}

    # --------------------- (1) warm start ---------------------
    print('warm start')
    B = B + random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    all_obj = []
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        print('latency: ', Y_latency[-1])
        all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L, soft=False))
    tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
    write_traj(V, c, B[tmp_opt_idx], 'solve_opt_active')

    # --------------------- (2) choose B ---------------------
    for i_epoches in tqdm(range(epoches)):

        # search 
        top_idx = np.argsort(all_obj)[::-1][:topK]
        B_top = list(np.array(B)[top_idx])
        B_0 = nearby_sample(n_model, B_top, B, n_samples=N3)

        # profile
        for b in B_0:
            # get_accuracy_profile
            Y_accuracy.append(get_accuracy_profile(V, b))
            # get_latency_profile
            tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
            all_latency.append(tmp_latency)
            Y_latency.append(np.percentile(tmp_latency, 95))
            all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L, soft=False))

        B = B + B_0
        print(np.array(B).shape)
        tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
        write_traj(V, c, B[tmp_opt_idx], 'solve_opt_active')

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
    opt_idx = np.argmax(np.nan_to_num(all_obj))
    opt_b = B[opt_idx]
    write_traj(V, c, opt_b, 'solve_opt_active')
    write_res(V, c, opt_b, 'solve_opt_active')

    return opt_b

##################################################################################################
# proxy
##################################################################################################
def solve_proxy(V, c, L, lamda):

    global opt_b_solve_random
    global opt_b_solve_greedy_accuracy
    global opt_b_solve_greedy_latency

    print("="*60)
    print("start solve_proxy")
    # --------------------- hyper parameters ---------------------
    
    if global_debug:
        N1 = 1 # warm start
        N2 = 10 # proxy
        N3 = 1 # profile
        epoches = 1
    else:
        N1 = 100 # warm start
        N2 = 1000 # proxy
        N3 = 30 # profile
        epoches = 10

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    try:
        B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
        print('warm init success', B)
    except:
        print('warm init failed')
        if global_debug:
            B = []
        else:
            opt_b_solve_random = solve_random(V, c, L, lamda)
            opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
            opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)
            B = [opt_b_solve_random, opt_b_solve_greedy_accuracy, opt_b_solve_greedy_latency]
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}
    accuracy_predictor = RF()
    latency_predictor = RF()

    # --------------------- (1) warm start ---------------------
    print('warm start')
    B = B + random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    all_obj = []
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        print('latency: ', Y_latency[-1])
        all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L, soft=False))
    tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
    write_traj(V, c, B[tmp_opt_idx], 'solve_proxy')

    # --------------------- (2) choose B ---------------------
    print('choose B start')
    for i_epoches in tqdm(range(epoches)):

        # fit proxy
        accuracy_predictor.fit(B, Y_accuracy)
        latency_predictor.fit(B, Y_latency)

        pred_accuracy = accuracy_predictor.predict(B)
        pred_latency = latency_predictor.predict(B)
        print(my_eval(Y_accuracy, pred_accuracy))
        print(my_eval(Y_latency, pred_latency))

        # search
        # random sample a large
        B_bar = random_sample(n_model=n_model, B=B, n_samples=N2)
        pred_accuracy = accuracy_predictor.predict(B_bar)
        pred_latency = latency_predictor.predict(B_bar)
        all_obj = []
        for i in range(len(B_bar)):
            all_obj.append(get_obj(pred_accuracy[i], pred_latency[i], lamda, L, soft=False))
        top_idx = np.argsort(all_obj)[::-1][:N3]
        B_0 = list(np.array(B_bar)[top_idx])

        # profile
        for b in tqdm(B_0):
            # get_accuracy_profile
            Y_accuracy.append(get_accuracy_profile(V, b))
            # get_latency_profile
            tmp_latency = get_latency_profile(V, c, b, cache=cache_latency)
            all_latency.append(tmp_latency)
            Y_latency.append(np.percentile(tmp_latency, 95))
            print('latency: ', Y_latency[-1])

        B = B + B_0
        print(np.array(B).shape)
        all_obj = []
        for i in range(len(B)):
            all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
        tmp_opt_idx = np.argmax(np.nan_to_num(all_obj))
        write_traj(V, c, B[tmp_opt_idx], 'solve_proxy')

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L, soft=False))
    opt_idx = np.argmax(np.nan_to_num(all_obj))
    opt_b = B[opt_idx]
    write_traj(V, c, opt_b, 'solve_proxy')
    write_res(V, c, opt_b, 'solve_proxy')

    return opt_b

if __name__ == "__main__":

    global_debug = False
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_latency = read_cache()
    log_name = 'res/log_{}.txt'.format(current_time)
    traj_name = 'res/traj_{}.txt'.format(current_time)

    if global_debug:
        L = 1.0 # maximum latency
    else:
        L = 1.5 # maximum latency
    lamda = 1
    n_patients_list = [1] # [1,2,5,10,20,50,100]
    
    with open(log_name, 'w') as fout:
        fout.write(current_time+'\n')
    
    for n_patients in n_patients_list:

        V, c = get_description(n_gpu=1, n_patients=n_patients)
        print('model description:', V, '\nsystem description:', c)

        # ---------- naive solutions ----------
        opt_b_solve_random = solve_random(V, c, L, lamda)
        opt_b_solve_greedy_accuracy = solve_greedy_accuracy(V, c, L, lamda)
        opt_b_solve_greedy_latency = solve_greedy_latency(V, c, L, lamda)

        # ---------- opt solutions ----------
        opt_b_solve_opt_passive = solve_opt_passive(V, c, L, lamda)
        opt_b_solve_opt_active = solve_opt_active(V, c, L, lamda)

        # ---------- proxy solutions ----------
        opt_b_solve_proxy = solve_proxy(V, c, L, lamda)


