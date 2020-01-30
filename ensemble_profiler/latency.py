from ray.experimental import serve
import os
import ray
from ensemble_profiler.utils import *
import time
from ensemble_profiler.server import HTTPActor
import subprocess
from ensemble_profiler.constants import ROUTE_ADDRESS, PROFILE_ENSEMBLE
import time
from threading import Event

package_directory = os.path.dirname(os.path.abspath(__file__))


def profile_ensemble(model_list, file_path, num_patients=1,
                     http_host="0.0.0.0", fire_clients=True,
                     with_data_collector=False):
    if not ray.is_initialized():
        serve.init(blocking=True, http_port=5000)
        nursery_handle = start_nursery()
        if not os.path.exists(str(file_path.resolve())):
            file_path.touch()
        file_name = str(file_path.resolve())

        # create the pipeline
        pipeline = create_services(model_list)

        # create patient handles
        if with_data_collector:
            actor_handles = start_patient_actors(num_patients=num_patients,
                                                 nursery_handle=nursery_handle,
                                                 pipeline=pipeline)
        else:
            actor_handles = {f"patient{i}": None for i in range(num_patients)}

        # start the http server
        obj_id = nursery_handle.start_actor.remote(HTTPActor,
                                                   "HEALTH_HTTP_SERVER",
                                                   init_args=[ROUTE_ADDRESS,
                                                              actor_handles,
                                                              pipeline,
                                                              file_name])
        http_actor_handle = ray.get(obj_id)[0]
        http_actor_handle.run.remote(host=http_host, port=8000)
        # wait for http actor to get started
        time.sleep(2)

        # fire client
        if fire_clients:
            print("Firing the clients")
            if with_data_collector:
                client_path = os.path.join(
                    package_directory, "patient_client.go")
                cmd = ["go", "run", client_path]
            else:
                ensembler_path = os.path.join(
                    package_directory, "profile_ensemble.go")
                cmd = ["go", "run", ensembler_path]
            # patient_name]
            procs = []
            for patient_name in actor_handles.keys():
                final_cmd = cmd + [patient_name]
                ls_output = subprocess.Popen(final_cmd)
                procs.append(ls_output)
            for p in procs:
                p.wait()
            serve.shutdown()
        else:
            # while True:
            #     time.sleep(1)
            Event().wait()
            serve.shutdown()
