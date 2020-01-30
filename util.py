import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from evaluate_results import evaluate_ensemble_models
from pathlib import Path
import os
import json


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

def get_accuracy_profile(V, b, debug=False):
    """
    """
    if debug:
        return np.random.rand()
    auc = evaluate_ensemble_models(b, num_record_eval=100, debug=True)
    return auc


def get_latency_profile(V, c, b, debug=False):
    """
    """
    if debug:
        return 1e-3*np.random.rand(100)
    v = V[np.array(b, dtype=bool)]
    print(V.shape, v.shape, b)
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
    profiler.profile_ensemble(model_list,file_path,system_constraint)

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

    return latency

