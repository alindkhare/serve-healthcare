import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from evaluate_results import evaluate_ensemble_models, evaluate_ensemble_models_per_patient
from pathlib import Path
import os
import json
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_model(base_filters, n_block):
    model = ResNet1D(in_channels=1,
                    base_filters=base_filters,
                    kernel_size=16,
                    stride=2,
                    n_block=n_block,
                    groups=base_filters,
                    n_classes=2,
                    downsample_gap=max(n_block//8, 1),
                    increasefilter_gap=max(n_block//4, 1),
                    verbose=False)
    # print(model.get_info())
    return model

def my_eval(gt, pred):
    return sqrt(mean_squared_error(gt, pred))

def dist(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))

def get_accuracy_profile(V, b, return_all=False):
    """
    """

    if return_all:
        if np.sum(b) == 0:
            return 0,0,0,0,0,0,0,0,0,0
        else:
            roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std = evaluate_ensemble_models_per_patient(b)
            return roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std
    else:
        if np.sum(b) == 0:
            return 0
        else:
            try:
                roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1_score,f1_score_std,precision,precision_std,recall,recall_std = evaluate_ensemble_models_per_patient(b)
                return roc_auc
            except:
                print(b)
                return 0
            


def get_latency_profile(V, c, b, cache, debug=False):
    """
    """
    print('profiling: ', b)

    if debug:
        return 1e-3*np.random.rand(100)
    if np.sum(b) == 0:
        return 1e6

    for i in cache:
        if dist(b, i[0]) == 0:
            print('using cache!')
            return i[1]

    v = V[np.array(b, dtype=bool)]
    model_list = []
    for i_model in v:
        model_list.append(get_model(int(i_model[0]), int(i_model[1])))

    filename = "profile_results.jsonl"
    p = Path(filename)
    p.touch()
    os.environ["SERVE_PROFILE_PATH"] = str(p.resolve())
    file_path = Path(filename)
    system_constraint = {"gpu":int(c[0]), "npatient":int(c[1])}
    print(system_constraint)
    profiler.profile_ensemble(model_list, file_path, system_constraint, fire_clients=False, with_data_collector=False)

    input_file = filename
    json_list = []
    df = pd.DataFrame(columns=['request_arrival','latency(ms)'])
    with open(input_file) as f:
        for line in f:
            json_list.append(json.loads(line))
    #print(json_list)
    for i, item in enumerate(json_list):
        latency = item["end"] - item["start"]
        df = df.append({'queue_id': i,'latency':latency}, ignore_index=True)
    print("res shape: ", df.shape)
    #print("sort list {}".format(sorted(df["latency"])))
    latency = df["latency"]

    cache.append([list(b),list(latency.values)])
    with open('cache_latency.txt', 'a') as fout:
        fout.write('{},{}\n'.format(list(b), list(latency.values)))

    return latency

def read_cache():
    fname = 'cache_latency.txt'
    cache_latency = []
    with open(fname, 'r') as fin:
        fin.readline()
        for line in fin:
            content = line.strip('\n|[|]').split('],[')
            b = np.array([int(float(i)) for i in content[0].split(',')])
            latency = np.array([float(i) for i in content[1].split(',')])
            cache_latency.append([b, np.percentile(latency, 95), latency])
    return cache_latency

def get_description(n_gpu, n_patients):
    """
    return V and c

    V: base_filters, n_block, accuracy, flops, size
    c: n_gpu, n_patients
    """

    df = pd.read_csv('model_list_small.csv')
    V = df.iloc[:, 1:].values
    c = np.array([n_gpu, n_patients])

    return V, c

def test_fit_latency():

    cache_latency = read_cache()
    B = []
    latency = []
    for i in cache_latency:
        B.append(i[0])
        latency.append(i[1])

    B = np.array(B)
    latency = np.array(latency)

    X_train, X_test, y_train, y_test = train_test_split(B, latency, test_size=0.9, random_state=0)
    print(X_train.shape, X_test.shape)

    m = RF()
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(mean_absolute_error(y_test, pred))
    print(r2_score(y_test, pred))


    plt.figure(figsize=(4,3))
    plt.plot([0,2],[0,2], 'r--')
    plt.scatter(y_test, pred, c='tab:grey', s=2)
    plt.xlabel('True Latency')
    plt.ylabel('Predicted Latency')
    plt.tight_layout()
    plt.savefig('img/cor_latency.png')

def plot_accuracy_latency():

    V, c = get_description(n_gpu=1, n_patients=1)
    cache_latency = read_cache()
    B = []
    latency = []
    accuracy = []
    step = 0
    for i in tqdm(cache_latency):
        step += 1
        B.append(i[0])
        latency.append(i[1])
        accuracy.append(get_accuracy_profile(V, i[0]))

        if step % 100 == 0:
            plt.figure(figsize=(4,3))
            plt.scatter(accuracy, latency, c='tab:grey', s=2)
            plt.xlabel('Accuracy')
            plt.ylabel('Latency')
            plt.tight_layout()
            plt.savefig('img/accuracy_2_latency.png')


if __name__ == "__main__":

    # fname = 'cache_latency.txt'
    # fname1 = 'cache_latency1.txt'
    # cache_latency = []
    # with open(fname, 'r') as fin, open(fname1, 'w') as fout:
    #     for line in fin:
    #         fout.write(line[:79]+', 0.0, 0.0, 0.0, 0.0'+line[79:])

    plot_accuracy_latency()
            