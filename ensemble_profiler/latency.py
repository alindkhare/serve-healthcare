from ray.experimental import serve
import os
import ray
from ensemble_profiler.utils import *
import time
from ensemble_profiler.server import HTTPActor
import subprocess
from ensemble_profiler.constants import ROUTE_ADDRESS


package_directory = os.path.dirname(os.path.abspath(__file__))


def profile_ensemble(model_list, file_path, num_patients=1,
                     http_host="0.0.0.0", fire_clients=True):
    if not ray.is_initialized():
        serve.init(blocking=True, http_port=5000)
        nursery_handle = start_nursery()
        if not os.path.exists(str(file_path.resolve())):
            file_path.touch()
        file_name = str(file_path.resolve())

        # create the pipeline
        pipeline = create_services(model_list)

        # create patient handles
        actor_handles = start_patient_actors(num_patients=num_patients,
                                             nursery_handle=nursery_handle,
                                             pipeline=pipeline)

        # start the http server
        # obj_id = nursery_handle.start_actor.remote(HTTPActor,
                                                   "HEALTH_HTTP_SERVER",
                                                   init_args = [ROUTE_ADDRESS,
                                                              actor_handles,
                                                              file_name])
        # http_actor_handle = ray.get(obj_id)[0]
        # http_actor_handle.run.remote(host=http_host, port=8000)
        # wait for http actor to get started
        http_server=HTTPActor(ROUTE_ADDRESS, actor_handles, file_name)
        http_server.run(host = http_host, port = 8000)
        # time.sleep(2)

        # # fire client
        # if fire_clients:
        #     client_path=os.path.join(package_directory, "patient_client.go")
        #     procs=[]
        #     for patient_name in actor_handles.keys():
        #         ls_output=subprocess.Popen(
        #             ["go", "run", client_path, patient_name])
        #         procs.append(ls_output)
        #     for p in procs:
        #         p.wait()
        #     serve.shutdown()
