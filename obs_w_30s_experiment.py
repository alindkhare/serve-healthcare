
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from pathlib import Path
import ray.experimental.serve as serve
import ray
import pandas as pd 
import jsonlines

# ECG
n_channel = 1
base_filters = 128
kernel_size = 16
n_classes = 2
n_block = 4
model = ResNet1D(in_channels=n_channel,
                 base_filters=base_filters,
                 kernel_size=kernel_size,
                 stride=2,
                 n_block=n_block,
                 groups=base_filters,
                 n_classes=n_classes,
                 downsample_gap=max(n_block//8, 1),
                 increasefilter_gap=max(n_block//4, 1),
                 verbose=False)

all_exp_lat = {}
for i in range(5):
    obs_w_30sec = pow(2,i)
    npatient = 1
    print("find all latency profile for {} patients".format(npatient))
    filename = "exp_window_{}patient_{}obs_w_30sec.jsonl".format(npatient, obs_w_30sec)
    file_path = Path(filename)
    constraint = {"gpu":1, "npatient":npatient}
    profile_lat = []
    exp_count = 10
    for i in range(exp_count):
        final_latency = profiler.profile_ensemble([model], file_path, constraint, fire_clients=True, with_data_collector=False, obs_w_30sec=obs_w_30sec)
        profile_lat.append(final_latency)
        print("latency_95th_profile-{} profiled: {} s".format(i, final_latency))
        key = "obs_w_30sec_{}".format(obs_w_30sec)
        all_exp_lat.update({key : profile_lat})
    df = pd.DataFrame.from_dict(all_exp_lat, orient='index').transpose()
    df.to_csv("1patient_obs_w_30sec_lat.csv",index=False)

