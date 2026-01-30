'''
Monkey-patching elastic code
'''
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm_scale.loader import load_model, zero_copy_model
DefaultModelLoader.load_model = load_model
DefaultModelLoader.zero_copy_model = zero_copy_model

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_scale.gpu_model_runner import get_supported_tasks
GPUModelRunner.get_supported_tasks = get_supported_tasks

from vllm_ascend.worker.worker import NPUWorker
from vllm_scale.npu_worker import reload_model
NPUWorker.reload_model = reload_model


## These need to be patched afterwards to avoid circular import
def patch_post_init():
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm_scale.scale.async_llm import reload_models, reload_kvcache
    AsyncLLM.reload_models = reload_models
    AsyncLLM.reload_kvcache = reload_kvcache

    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm_scale.scale.core_client import (reload_models_async, reload_kvcache_async)
    AsyncMPClient.reload_models_async = reload_models_async
    AsyncMPClient.reload_kvcache_async = reload_kvcache_async

    from vllm.v1.engine.core import EngineCore
    from vllm_scale.scale.core import reload_model, reload_kv_cache
    EngineCore.reload_model = reload_model
    EngineCore.reload_kv_cache = reload_kv_cache

    from vllm.v1.executor.abstract import Executor
    from vllm_scale.scale.abstract import reload_model, reload_kv_cache
    Executor.reload_model = reload_model
    Executor.reload_kv_cache = reload_kv_cache

