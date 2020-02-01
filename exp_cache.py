import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import pickle
import random
from tqdm import tqdm
from datetime import datetime

from util import my_eval, get_accuracy_profile, get_latency_profile
from exp_autoscale import get_description

if __name__ == "__main__":

    outname = 'cache_latency.txt'
    V, c = get_description(n_gpu=2, n_patients=100)
    n_model = V.shape[0]

    # 1 model
    for i1 in range(n_model):
        b = np.zeros(n_model)
        b[i1] = 1
        tmp_latency = get_latency_profile(V, c, b)
        with open(outname, 'a') as fout:
            fout.write('{},{}\n'.format(b, tmp_latency))

    # 2 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model):
            b = np.zeros(n_model)
            b[i1] = 1
            b[i2] = 1
            tmp_latency = get_latency_profile(V, c, b)
            with open(outname, 'a') as fout:
                fout.write('{},{}\n'.format(b, tmp_latency))

    # 3 models
    for i1 in range(n_model-1):
        for i2 in range(i1+1, n_model-1):
            for i3 in range(i2+1, n_model):
                b = np.zeros(n_model)
                b[i1] = 1
                b[i2] = 1
                b[i3] = 1
                tmp_latency = get_latency_profile(V, c, b)
                with open(outname, 'a') as fout:
                    fout.write('{},{}\n'.format(b, tmp_latency))




