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
import requests
import json
import socket
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


def profile_ensemble(model_list, file_path, constraint={"gpu":1, "npatient":1},
                     http_host="0.0.0.0", fire_clients=True, with_data_collector=False):
    if not ray.is_initialized():
        #read constraint
        num_patients = int(constraint["npatient"])
        gpu = int(constraint["gpu"])
        serve.init(blocking=True, http_port=5000)
        nursery_handle = start_nursery()
        if not os.path.exists(str(file_path.resolve())):
            file_path.touch()
        file_name = str(file_path.resolve())
        # create the pipeline
        pipeline = create_services(model_list,gpu)
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
            gw = os.popen("ip -4 route show default").read().split()
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((gw[2], 0))
            IPv4addr = s.getsockname()[0]  #for where the server ray.serve() request will be executed
            serve_port = 8000

            url = "http://130.207.25.143:4000/jsonrpc" #for client address. In the experiment points to pluto
            print("sending RPC request form IPv4 addr: {}".format(IPv4addr))
            if with_data_collector:
                req_params = {"npatient":num_patients, "serve_ip":IPv4addr, "serve_port":serve_port, "go_client_name":"patient_client"}
            else:
                req_params = {"npatient":num_patients, "serve_ip":IPv4addr, "serve_port":serve_port, "go_client_name":"profile_ensemble"}
            fire_remote_clients(url, req_params)
            print("finish firing remote clients")
            serve.shutdown()

def fire_remote_clients(url, req_params):
    payload = {
        "method": "fire_client",
        "params": req_params,
        "jsonrpc": "2.0",
        "id": 0
    }
    response = requests.post(url, json=payload).json()
    print("{}".format(response))

