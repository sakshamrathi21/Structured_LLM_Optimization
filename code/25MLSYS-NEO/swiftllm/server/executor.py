"""
Model executor classes.

Provides control plane APIs for the engine. Calls the data plane APIs under the hood.
"""

import os
from abc import ABC, abstractmethod

import ray

from swiftllm.worker.model import ModelPerfResult, LlamaModel, RemoteLlamaModel
from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig

class Executor(ABC):
    """
    Base class for executors.
    """
    def __init__(
        self, 
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        raise NotImplementedError

    
    @abstractmethod
    def init_kvcache_and_swap(self):
        """
        Initialize the key-value cache and swap.
        """
        raise NotImplementedError

    
    @abstractmethod
    def do_one_iteration(self, *args) -> list[int]:
        """
        Do one iteration of the model.
        """
        raise NotImplementedError


    @abstractmethod
    def turn_on_perf_monitor(self):
        """
        Turn on performance monitoring.
        """
        raise NotImplementedError


    @abstractmethod
    def turn_off_perf_monitor_and_flush_results(self) -> list[ModelPerfResult]:
        """
        Turn off performance monitoring and flush results.
        """
        raise NotImplementedError



class SingleProcExecutor(Executor):
    """
    Single process executor.
    """
    def __init__(
        self, 
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        self.engine_config = engine_config
        self.model_config = model_config
        tpd = engine_config.tensor_parallel_degree
        assert tpd == 1, f"SingleProcExecutor does not support tensor parallelism degree({tpd}) == 1"
        self.model = LlamaModel(engine_config, model_config, rank=0)

    
    def init_kvcache_and_swap(self):
        self.model.init_kvcache_and_swap(self.engine_config)

    
    def do_one_iteration(self, *args) -> list[int]:
        return self.model.do_one_iteration(*args)

    
    def turn_on_perf_monitor(self):
        self.model.turn_on_perf_monitor()


    def turn_off_perf_monitor_and_flush_results(self) -> list[ModelPerfResult]:
        return self.model.turn_off_perf_monitor_and_flush_results()


class RayExecutor(Executor):
    """
    Ray executor. Inits ray framework when instantiated.
    """
    # pylint: disable=no-member
    def __init__(
        self, 
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        self.engine_config = engine_config
        self.model_config = model_config

        num_workers = engine_config.tensor_parallel_degree
        self.models = [RemoteLlamaModel.remote(engine_config, model_config, rank=i) for i in range(num_workers)]
    
    
    def init_kvcache_and_swap(self):
        ray.get([model.init_kvcache_and_swap.remote(self.engine_config) for model in self.models])

    
    def do_one_iteration(self, *args) -> list[int]:
        return ray.get([model.do_one_iteration.remote(*args) for model in self.models])[0]

    
    def turn_on_perf_monitor(self):
        ray.get(self.models[0].turn_on_perf_monitor.remote())


    def turn_off_perf_monitor_and_flush_results(self) -> list[ModelPerfResult]:
        return ray.get(self.models[0].turn_off_perf_monitor_and_flush_results.remote())
