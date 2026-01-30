import os, torch
_LOADED = False

def ensure_ops_loaded():
    global _LOADED
    if _LOADED:
        return
    try:
        _ = torch.ops.tensor_ipc_utils.hello_word
        _LOADED = True
        return
    except (AttributeError, RuntimeError):
        pass
    lib = os.path.join(os.path.dirname(__file__), "tensor_ip_utils.cpython-310-aarch64-linux-gnu.so")
    torch.ops.load_library(lib)
    _LOADED = True