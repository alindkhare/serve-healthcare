import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile

def get_description(n_gpu, n_patients):
    """
    return V and c

    V: base_filters, n_block, accuracy, latency
    c: n_gpu, n_patients
    """

    base_filters_list = [8, 16, 32, 64, 128]
    n_block_list = [2, 4, 8, 16]
    n_fields = 3
    n_model = len(base_filters_list) * len(n_block_list)
    V = []
    for base_filters in base_filters_list:
        for n_block in n_block_list:
            accuracy = np.random.rand()
            latency = 1e-4*np.random.rand()
            tmp = [base_filters, n_block, accuracy, latency]
            V.append(tmp)
    V = np.array(V)

    n_gpu = n_gpu
    n_patients = n_patients
    c = np.array([n_gpu, n_patients])

    return V, c

def random_sample(n_model, B, n_samples):
    """
    Input:
        n_model: number of models, n
        n_samples: 

    Output:
        B \in \{0,1\}^{n_samples \times n_model}
    """
    out = []
    i = 0
    while i < n_samples:
        # get a random probability of 1s and 0s
        pp = np.random.rand()
        # get random binary vector
        tmp = np.random.choice([0, 1], size=n_model, p=(pp,1-pp))
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

def get_obj(accuracy, latency, lamda, L):
    return accuracy + lamda * (L - latency)

def save_checkpoint(res):
    with open('res.pkl','wb') as fout:
        pickle.dump(res, fout)

##################################################################################################
# naive
##################################################################################################
def solve_random(V, c, L, lamda):
    """
    random incremental
    """
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_all = list(range(n_model))
    latency = 0.0
    for i in range(n_model):
        tmp_idx = np.random.choice(idx_all)
        idx_all.remove(tmp_idx)
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b)
        latency = np.percentile(tmp_latency, 95)
        print(tmp_idx, b, latency)
        if latency >= L:
            break
    print('found best b is: {}'.format(b))
    return b

def solve_greedy_accuracy(V, c, L, lamda):
    """
    greedy accuracy incremental
    """
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 2])[::-1] # the 3rd col is accuracy
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b)
        latency = np.percentile(tmp_latency, 95)
        print(tmp_idx, b, latency)
        if latency >= L:
            break
    print('found best b is: {}'.format(b))
    return b

def solve_greedy_latency(V, c, L, lamda):
    """
    greedy latency incremental
    """
    n_model = V.shape[0]
    b = np.zeros(n_model)
    idx_order = np.argsort(V[:, 3]) # the 4th col is latency
    latency = 0.0
    for i in range(n_model):
        tmp_idx = idx_order[i]
        b[tmp_idx] = 1
        tmp_latency = get_latency_profile(V, c, b)
        latency = np.percentile(tmp_latency, 95)
        print(tmp_idx, b, latency)
        if latency >= L:
            break
    print('found best b is: {}'.format(b))
    return b

##################################################################################################
# opt
##################################################################################################
def solve_opt_passive(V, c, L, lamda):
    # --------------------- hyper parameters ---------------------
    
    N1 = 100 # warm start

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    B = []
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}

    # --------------------- (1) warm start ---------------------
    B = random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        save_checkpoint(res)

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L))
    opt_idx = np.argmax(all_obj)
    opt_b = B[opt_idx]
    print('found best b is: {}'.format(opt_b))
    return opt_b

def solve_opt_active(V, c, L, lamda):
    # --------------------- hyper parameters ---------------------
    
    N1 = 10 # warm start
    topK = 3 # search near top
    N3 = 10 # profile
    epoches = 10

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    B = []
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}

    # --------------------- (1) warm start ---------------------
    B = random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    all_obj = []
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        save_checkpoint(res)
        all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L))

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
            tmp_latency = get_latency_profile(V, c, b)
            all_latency.append(tmp_latency)
            Y_latency.append(np.percentile(tmp_latency, 95))
            save_checkpoint(res)
            all_obj.append(get_obj(Y_accuracy[-1], Y_latency[-1], lamda, L))

        B = B + B_0
        print(np.array(B).shape)

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L))
    opt_idx = np.argmax(all_obj)
    opt_b = B[opt_idx]
    print('found best b is: {}'.format(opt_b))
    return opt_b

##################################################################################################
# proxy
##################################################################################################
def solve_proxy(V, c, L, lamda):
    # --------------------- hyper parameters ---------------------
    
    N1 = 10 # warm start
    N2 = 1000 # proxy
    N3 = 10 # profile
    epoches = 10

    # --------------------- initialization ---------------------
    n_model = V.shape[0]
    B = []
    Y_accuracy = []
    all_latency = []
    Y_latency = []
    res = {'B':B, 'Y_accuracy':Y_accuracy, 'Y_latency':Y_latency, 'all_latency':all_latency}
    accuracy_predictor = RF()
    latency_predictor = RF()

    # --------------------- (1) warm start ---------------------
    print('warm start')
    B = random_sample(n_model=n_model, B=B, n_samples=N1)
    # profile
    for b in tqdm(B):
        Y_accuracy.append(get_accuracy_profile(V, b))
        tmp_latency = get_latency_profile(V, c, b)
        all_latency.append(tmp_latency)
        Y_latency.append(np.percentile(tmp_latency, 95))
        save_checkpoint(res)

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
            all_obj.append(get_obj(pred_accuracy[i], pred_latency[i], lamda, L))
        top_idx = np.argsort(all_obj)[::-1][:N3]
        B_0 = list(np.array(B_bar)[top_idx])

        # profile
        for b in tqdm(B_0):
            # get_accuracy_profile
            Y_accuracy.append(get_accuracy_profile(V, b))
            # get_latency_profile
            tmp_latency = get_latency_profile(V, c, b)
            all_latency.append(tmp_latency)
            Y_latency.append(np.percentile(tmp_latency, 95))
            save_checkpoint(res)

        B = B + B_0
        print(np.array(B).shape)

    # --------------------- (3) solve ---------------------
    all_obj = []
    for i in range(len(B)):
        all_obj.append(get_obj(Y_accuracy[i], Y_latency[i], lamda, L))
    opt_idx = np.argmax(all_obj)
    opt_b = B[opt_idx]
    print('found best b is: {}'.format(opt_b))
    return opt_b

if __name__ == "__main__":
    
    L = 0.1 # maximum latency
    lamda = 10
    V, c = get_description(n_gpu=4, n_patients=1)

    # ---------- naive solutions ----------
    solve_random(V, c, L, lamda)
    # solve_greedy_accuracy(V, c, L, lamda)
    # solve_greedy_latency(V, c, L, lamda)

    # # ---------- opt solutions ----------
    # solve_opt_passive(V, c, L, lamda)
    # solve_opt_active(V, c, L, lamda)

    # # ---------- proxy solutions ----------
    # solve_proxy(V, c, L, lamda)
