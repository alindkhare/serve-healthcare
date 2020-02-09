import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm

from util import my_eval, get_accuracy_profile, get_latency_profile, get_description, get_description_small, get_now, b2cnt, cnt2b, read_cache_latency

def b_cache():

    outname = 'cache_latency.txt'
    # with open(outname, 'w') as fout:
    #     fout.write(get_now()+'\n')
    
    V, c = get_description_small(n_gpu=1, n_patients=1)

    # 1 model
    for i1 in range(n_model):
        b = np.zeros(n_model)
        b[i1] = 1
        tmp_latency = get_latency_profile(V, c, b, cache=None)
        with open(outname, 'a') as fout:
            fout.write('{},{},{}\n'.format(get_now(), list(b), tmp_latency))

    # 2 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model):
            b = np.zeros(n_model)
            b[i1] = 1
            b[i2] = 1
            tmp_latency = get_latency_profile(V, c, b, cache=None)
            with open(outname, 'a') as fout:
                fout.write('{},{},{}\n'.format(get_now(), list(b), tmp_latency))

    # 3 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model-1):
            for i3 in range(i2+1, n_model):
                b = np.zeros(n_model)
                b[i1] = 1
                b[i2] = 1
                b[i3] = 1
                tmp_latency = get_latency_profile(V, c, b, cache=None)
                with open(outname, 'a') as fout:
                    fout.write('{},{},{}\n'.format(get_now(), list(b), tmp_latency))

    # 4 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model-1):
            for i3 in range(i2+1, n_model-1):
                for i4 in range(i3+1, n_model):
                    b = np.zeros(n_model)
                    b[i1] = 1
                    b[i2] = 1
                    b[i3] = 1
                    b[i4] = 1
                    tmp_latency = get_latency_profile(V, c, b, cache=None)
                    with open(outname, 'a') as fout:
                        fout.write('{},{},{}\n'.format(get_now(), list(b), tmp_latency))

    # 5 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model-1):
            for i3 in range(i2+1, n_model-1):
                for i4 in range(i3+1, n_model-1):
                    for i5 in range(i4+1, n_model):
                        b = np.zeros(n_model)
                        b[i1] = 1
                        b[i2] = 1
                        b[i3] = 1
                        b[i4] = 1
                        b[i5] = 1
                        tmp_latency = get_latency_profile(V, c, b, cache=None)
                        with open(outname, 'a') as fout:
                            fout.write('{},{},{}\n'.format(get_now(), list(b), tmp_latency))

def cnt_cache(cache):

    outname = 'cache_latency.txt'
    V, c = get_description_small(n_gpu=1, n_patients=1)

    # 1 model
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    cnt = [0]*16 + [i1, i2, i3, i4]
                    b = cnt2b(cnt, V)
                    tmp_latency = get_latency_profile(V, c, b, cache=cache)

if __name__ == "__main__":

    cache_latency = read_cache_latency()
    cnt_cache(cache_latency)



