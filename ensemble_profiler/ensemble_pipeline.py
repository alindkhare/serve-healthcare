import ray
import ray.experimental.serve as serve
from ensemble_profiler.constants import (SERVICE_STORE_ECG_DATA,
                                         AGGREGATE_PREDICTIONS)


class EnsemblePipeline:
    def __init__(self, model_services, service_handles):
        self.model_services = model_services
        self.service_handles = service_handles

    def remote(self, *args, **kwargs):
        ecg_store_object_id = ray.ObjectID.from_random()
        ecg_predicate_object_id = ray.ObjectID.from_random()
        self.service_handles[SERVICE_STORE_ECG_DATA].remote(
            *args, **kwargs, return_object_ids={
                serve.RESULT_KEY: ecg_store_object_id,
                serve.PREDICATE_KEY: ecg_predicate_object_id
            }
        )
        kwargs_for_aggregate = {}
        for model_service in self.model_services:
            md_object_id = ray.ObjectID.from_random()
            kwargs_for_aggregate[model_service] = md_object_id
            self.service_handles[model_service].remote(
                data = ecg_store_object_id,
                predicate_condition = ecg_predicate_object_id,
                default_value = ("kwargs","data"),
                return_object_ids = {serve.RESULT_KEY: md_object_id}
            )
        aggregate_object_id = ray.ObjectID.from_random()
        self.service_handles[AGGREGATE_PREDICTIONS].remote(
            **kwargs_for_aggregate,
            predicate_condition= ecg_predicate_object_id,
            default_value = ("kwargs",model_service[0]),
            return_object_ids={serve.RESULT_KEY: aggregate_object_id}
        )
        return aggregate_object_id
        
