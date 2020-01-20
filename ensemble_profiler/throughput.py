from ensemble_profiler.utils import create_services, start_patient_actors
import ray
from ray.experimental import serve


def calculate_throughput(model_list, num_queries=300):
    serve.init(blocking=True)
    pipeline = create_services(model_list)

    actor_handles = start_patient_actors(num_patients=1, pipeline=pipeline)
    patient_handle = list(actor_handles.values())[0]

    future_list = []

    # dummy request
    info = {
        "patient_name": PATIENT_NAME_PREFIX + str(0),
        "value": 1.0,
        "vtype": "ECG"
    }
    start_time = time.time()
    for _ in range(num_queries):
        fut = patient_handle.get_periodic_predictions.remote(info=info)
        future_list.append(fut)
    ray.get(future_list)
    end_time = time.time()
    serve.shutdown()
    return end_time - start_time, num_queries
