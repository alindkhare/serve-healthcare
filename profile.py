from resnet1d.resnet1d import ResNet1D
import ray.experimental.serve as serve
from store_data import StorePatientData
from patient_prediction import PytorchPredictorECG
from ray.experimental.serve import BackendConfig
import subprocess
from pathlib import Path
import os

# ECG
n_channel = 1
base_filters = 64
kernel_size = 16
n_classes = 2
n_block = 2
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

cuda = False
if cuda:
    hw = 'gpu'
else:
    hw = 'cpu'
# initiate serve
p = Path("Resnet1d_base_filters={},kernel_size={},n_block={}"
         "_{}_7600_queries_ray_serve.jsonl".format(base_filters, kernel_size, n_block, hw))
p.touch()
os.environ["SERVE_PROFILE_PATH"] = str(p.resolve())
serve.init(blocking=True)

# Kwargs creator for profiling the service
kwargs_creator = lambda : {'info': {"patient_name": "Adam",
                                    "value": 0.0,
                                    "vtype": "ECG"
                                    }
                          }

# create ECG service
serve.create_endpoint("ECG")
# create data point service for hospital
serve.create_endpoint("hospital", route="/hospital",
                      kwargs_creator=kwargs_creator)

# create backend for ECG
b_config = BackendConfig(num_replicas=1)
serve.create_backend(PytorchPredictorECG, "PredictECG",
                     model, cuda, backend_config=b_config)
# link service and backend
serve.link("ECG", "PredictECG")
handle = serve.get_handle("ECG")

# prepare args for StorePatientData backend.
service_handles_dict = {"ECG": handle}
# do prediction after every 3750 queries.
num_queries_dict = {"ECG": 3750}
# Always keep num_replicas as 1 as this is a stateful Backend
# This backend will store all the patient's data and transfer
# the prediction to respective Backend (ECG handle in this case)
b_config_hospital = BackendConfig(num_replicas=1)
serve.create_backend(StorePatientData, "StoreData",
                     service_handles_dict, num_queries_dict,
                     backend_config=b_config_hospital)
serve.link("hospital", "StoreData")
print("Started client!")
# fire client
procs = []
for _ in range(1):
    ls_output = subprocess.Popen(["go", "run", "patient_client.go"])
    procs.append(ls_output)
for p in procs:
    p.wait()
