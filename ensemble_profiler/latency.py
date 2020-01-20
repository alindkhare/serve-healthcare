from ray.experimental import serve
import os
import ray
from ensemble_profiler.utils import create_services, start_patient_actors
import time


def profile_ensemble(model_list, file_path, num_patients=1):
    serve.init(blocking=True)
    if not os.path.exists(str(file_path.resolve())):
        file_path.touch()
    file_name = str(file_path.resolve())

    # create the pipeline
    pipeline = _create_services(model_list)

    # create patient handles
    actor_handles = _start_patient_actors(num_patients=num_patients,
                                          pipeline=pipeline)

    # start the http server
    http_actor_handle = HTTPActor.remote(ROUTE_ADDRESS, actor_handles,
                                         file_name)
    http_actor_handle.run.remote()
    # wait for http actor to get started
    time.sleep(2)

    # fire client
    client_path = os.path.join(package_directory, "patient_client.go")
    procs = []
    for patient_name in actor_handles.keys():
        ls_output = subprocess.Popen(["go", "run", client_path, patient_name])
        procs.append(ls_output)
    for p in procs:
        p.wait()
    serve.shutdown()
