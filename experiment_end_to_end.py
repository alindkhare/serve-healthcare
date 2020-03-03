
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from pathlib import Path
import ray.experimental.serve as serve
import ray
import pandas as pd
# 1,1,solve_random,0.9639,0.0470,0.9315,0.0873,0.3316,0.2943,0.7476,0.4316,0.2497,0.2954,0.7498,0.0985,0.57698540,[0. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 1.]
# 1,1,solve_greedy_accuracy,0.9608,0.0431,0.9336,0.0734,0.5559,0.3494,0.8912,0.2739,0.4697,0.3542,0.8225,0.1180,0.51610991,[0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 1.]
# 1,1,solve_greedy_latency,0.9640,0.0549,0.9350,0.1059,0.4779,0.3391,0.7625,0.4054,0.3821,0.3120,0.7936,0.1042,0.54406964,[1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 0.]
# 1,1,solve_opt_passive,0.9787,0.0256,0.9540,0.0633,0.4919,0.3394,0.8830,0.2919,0.4021,0.3437,0.8003,0.1147,0.51318176,[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
# 1,1,solve_proxy,0.9792,0.0282,0.9573,0.0600,0.5845,0.3688,0.9030,0.1996,0.5051,0.3589,0.8344,0.1198,0.44609477,[1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0.]
def generate_resnet(base_filters = 128, n_block = 4):
    n_channel = 1 
    kernel_size = 16 
    n_classes = 2 
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
     
    return model

def solve_random():
    m_list = [generate_resnet(n_block=4), generate_resnet(n_block=16), generate_resnet(n_block=2), generate_resnet(n_block=4),
            generate_resnet(n_block=16), generate_resnet(n_block=4), generate_resnet(n_block=16)]
    return m_list

def solve_GA():
    m_list = [generate_resnet(n_block=16), generate_resnet(n_block=16), generate_resnet(n_block=8),
            generate_resnet(n_block=16)]
    return m_list

def solve_GL():
    m_list = [generate_resnet(n_block=2), generate_resnet(n_block=4), generate_resnet(n_block=8),
            generate_resnet(n_block=2), generate_resnet(n_block=4), generate_resnet(n_block=8),
            generate_resnet(n_block=2), generate_resnet(n_block=4), generate_resnet(n_block=8)]
    return m_list

def solve_opt():
    m_list = [generate_resnet(n_block=4), generate_resnet(n_block=8),
            generate_resnet(n_block=4), generate_resnet(n_block=8),
            generate_resnet(n_block=4), generate_resnet(n_block=16)]
    return m_list

def solve_opt_final():
    m_list = [generate_resnet(n_block=4), generate_resnet(base_filters=64,n_block=2),generate_resnet(base_filters=32,n_block=2),generate_resnet(base_filters=32,n_block=4),
            generate_resnet(base_filters=16,n_block=2),generate_resnet(base_filters=16,n_block=4),generate_resnet(base_filters=16,n_block=8),
            generate_resnet(base_filters=8,n_block=2), generate_resnet(base_filters=8,n_block=8)]
    return m_list

def solve_proxy():
    m_list = [generate_resnet(n_block=2), generate_resnet(n_block=2), generate_resnet(n_block=8),
            generate_resnet(n_block=16), generate_resnet(n_block=4), generate_resnet(n_block=8)]
    return m_list

list_lat = []
experiment = 10
model_list = solve_opt_final()
for pat in range(1):
    npatient = 32
    gpu = 1
    print("start experiment with {} patients".format(npatient))
    print("start experiment with {} gpu".format(gpu))
    filename = "solve_final_gpu{}_npatient{}_profile_results.jsonl".format(gpu, npatient)
    file_path = Path(filename)
    constraint = {"gpu":gpu, "npatient":npatient}
    lat = profiler.profile_ensemble(model_list, file_path, constraint, fire_clients=False, with_data_collector=True)
