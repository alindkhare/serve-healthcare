import subprocess
from pathlib import Path
import os
from ray.experimental.serve import BackendConfig
import ray.experimental.serve as serve
import ray

from ensemble_profiler.constants import (MODEL_SERVICE_ECG_PREFIX,
                                         AGGREGATE_PREDICTIONS,
                                         BACKEND_PREFIX,
                                         ROUTE_ADDRESS,
                                         PATIENT_NAME_PREFIX)
from ensemble_profiler.store_data_actor import StatefulPatientActor
from ensemble_profiler.patient_prediction import PytorchPredictorECG
from ensemble_profiler.ensemble_predictions import Aggregate
from ensemble_profiler.ensemble_pipeline import EnsemblePipeline
from ensemble_profiler.server import HTTPActor
import time
package_directory = os.path.dirname(os.path.abspath(__file__))


def _create_services(model_list):
    all_services = []
    # create relevant services
    model_services = []
    for i in range(len(model_list)):
        model_service_name = MODEL_SERVICE_ECG_PREFIX + "::" + str(i)
        model_services.append(model_service_name)
        serve.create_endpoint(model_service_name)
    all_services += model_services
    serve.create_endpoint(AGGREGATE_PREDICTIONS)
    all_services.append(AGGREGATE_PREDICTIONS)

    for service, model in zip(model_services, model_list):
        b_config = BackendConfig(num_replicas=1, num_gpus=1)
        serve.create_backend(PytorchPredictorECG, BACKEND_PREFIX+service,
                             model, True, backend_config=b_config)
    serve.create_backend(Aggregate, BACKEND_PREFIX+AGGREGATE_PREDICTIONS)

    # link services to backends
    for service in all_services:
        serve.link(service, BACKEND_PREFIX+service)

    # get handles
    service_handles = {}
    for service in all_services:
        print("Services: {}".format(service))
        service_handles[service] = serve.get_handle(service)

    pipeline = EnsemblePipeline(model_services, service_handles)
    return pipeline


def _start_patient_actors(num_patients, pipeline, periodic_interval=3750):
    # start actor for collecting patients_data
    actor_handles = {}
    for patient_id in range(num_patients):
        patient_name = PATIENT_NAME_PREFIX + str(patient_id)
        handle = StatefulPatientActor.options(is_asyncio=False).remote(
            patient_name=patient_name,
            pipeline=pipeline,
            periodic_interval=periodic_interval
        )
        actor_handles[patient_name] = handle
    return actor_handles


def profile_ensemble(model_list, file_path, num_patients=1):
    serve.init(blocking=True)
    print("hahahahhaha")
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


def calculate_throughput(model_list, num_queries=20):
    serve.init(blocking=True)
    print("Call me")
    pipeline = _create_services(model_list)

    actor_handles = _start_patient_actors(num_patients=1, pipeline=pipeline)
    print(actor_handles)
    patient_handle = list(actor_handles.values())[0]
    print(patient_handle)

    future_list = []

    # dummy request
    info = {
        "patient_name": PATIENT_NAME_PREFIX + str(0),
        "value": 1.0,
        "vtype": "ECG"
    }
    # d = ray.get(patient_handle.get_periodic_predictions.remote(info=info))
    # print(d)
    # return
    start_time = time.time()
    for _ in range(num_queries):
        fut = patient_handle.get_periodic_predictions.remote(info=info)
        future_list.append(fut)
    result = ray.get(future_list)
    end_time = time.time()
    serve.shutdown()
    return end_time - start_time, num_queries
