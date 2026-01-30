import requests
import time
from concurrent.futures import ThreadPoolExecutor, wait
import argparse
import psutil
import socket

HEAD_IP = "localhost"
PORT = 8000
BASE = f"http://{HEAD_IP}:{PORT}"

def get_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    return r.json()

def get_model_health():
    r = requests.get(f"{BASE}/model_health", timeout=10)
    r.raise_for_status()
    return r.json()

def get_cluster_status():
    r = requests.get(f"{BASE}/cluster_status", timeout=10)
    r.raise_for_status()
    return r.json()

def post_broadcast_metadata():
    r = requests.post(f"{BASE}/broadcast_metadata", timeout=30)
    r.raise_for_status()
    return r.json()

def post_scaleup(num_npus=2):
    r = requests.post(f"{BASE}/scaleup", params={"num_npus": num_npus}, timeout=300)
    r.raise_for_status()
    return r.json()

def post_scaledown(num_npus=2):
    r = requests.post(f"{BASE}/scaledown", params={"num_npus": num_npus}, timeout=300)
    r.raise_for_status()
    return r.json()

def post_addnpus(num_npus=2):
    r = requests.post(f"{BASE}/addnpus", params={"num_npus": num_npus}, timeout=300)
    r.raise_for_status()
    return r.json()

def post_invoke_method(method_name: str, *args, **kwargs):
    payload = {}
    if args:
        payload["args"] = list(args)
    if kwargs:
        payload["kwargs"] = kwargs
    r = requests.post(
        f"{BASE}/invoke_method",
        params={"method_name": method_name},
        json=payload if payload else None,
        timeout=300
    )
    r.raise_for_status()
    return r.json()

def init_inference_engine(inference_port=7102):
    """wait until resources are available and initialize inference engine"""
    start_t = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        copy_model_res = executor.submit(
            requests.post, f"http://localhost:{inference_port}/reload_models"
        )
        copy_kv_res = executor.submit(
            requests.post, f"http://localhost:{inference_port}/reload_kvcache"
        )
        wait([copy_model_res, copy_kv_res])

    reload_t = time.time()
    print(f"zero-copy model & kv cache took {reload_t - start_t:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale up NPUs and initialize inference engine")
    parser.add_argument("--num_scale_units", type=int, default=0, help="Number of scale units (default: 1)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default: 2)")
    parser.add_argument("--inference_port", type=int, default=7101, help="Inference server port (default: 7101)")

    args = parser.parse_args()

    num_scale_units = args.num_scale_units
    TP = args.tp
    inference_port = args.inference_port

    ## Scale up 
    if num_scale_units > 0:
        print("scaleup:", post_scaleup(num_scale_units * TP))

    # Initialize inference engine
    scale_start_t = time.time()
    init_inference_engine(inference_port=inference_port)
    print(f"zero-copy time: {time.time() - scale_start_t:.2f} seconds")