
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from pathlib import Path
import ray.experimental.serve as serve
import ray

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

model_list = solve_opt_final()
npatient = 2
gpu = 1
print("start experiment with {} patients".format(npatient))
print("start experiment with {} gpu".format(gpu))
    
filename = "demo_gpu{}_npatient{}_profile_results.jsonl".format(gpu, npatient)
file_path = Path(filename)
constraint = {"gpu":gpu, "npatient":npatient}
lat = profiler.profile_ensemble(model_list, file_path, constraint, fire_clients=False, with_data_collector=True)
print("lat 95th: {}".format(lat))
