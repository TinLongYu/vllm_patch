import os

def reload_model(self) -> None:
    """
    Load weights by setting flag INFER_STATUS = 1
    """
    os.environ["INFER_STATUS"] = "1"
    self.model_executor.reload_model()

def reload_kvcache(self) -> None:
    """
    Load KVCache by setting flag INFER_STATUS = 1
    """
    os.environ["INFER_STATUS"] = "1"
    self._initialize_kv_caches(self.vllm_config)
