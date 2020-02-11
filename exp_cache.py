import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile, get_description, get_now, b2cnt, cnt2b, read_cache_latency

def cnt_cache_latency(cache):

    outname = 'cache_latency.txt'
    V, c = get_description(n_gpu=1, n_patients=1, is_small=True)

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    cnt = [0]*16 + [i1, i2, i3, i4]
                    b = cnt2b(cnt, V)
                    tmp_latency = get_latency_profile(V, c, b, cache=cache)

def b_cache_accuracy(cache):

    outname = 'cache_accuracy.txt'
    V, c = get_description(n_gpu=1, n_patients=1, is_small=True)

    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    cnt = [0]*16 + [i1, i2, i3, i4]
                    b = cnt2b(cnt, V)
                    final_res = get_accuracy_profile(V, b, cache=cache, return_all=True)

def precompute():

    V, c = get_description(n_gpu=1, n_patients=1, is_small=False)
    n_model = V.shape[0]
    for i in range(n_model):
        b = np.zeros(n_model)
        b[i] = 1
        tmp_accuracy = get_accuracy_profile(V, b, cache=None)
        tmp_latency = get_latency_profile(V, c, b, cache=None)
        print('{},{}'.format(tmp_accuracy, tmp_latency))

if __name__ == "__main__":

    cache_latency = read_cache_latency()
    cnt_cache_latency(cache=cache_latency)

    # cache_accuracy = read_cache_accuracy()
    # cnt_cache_accuracy(cache=cache_accuracy)

    # the biggest
    # V, c = get_description(n_gpu=1, n_patients=1, is_small=True)
    # b = [1] * 12
    # final_res = get_latency_profile(V, c, b, cache=None)

    precompute()

