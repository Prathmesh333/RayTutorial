import ray
from common import start_ray


@ray.remote(num_cpus=1, num_gpus=1)
def gpu_check():
    import torch
    assert torch.cuda.is_available(), "Use a compatible CUDA-enabled PyTorch build."
    x = torch.ones((16, 16), device="cuda")
    return {"device": torch.cuda.get_device_name(0), "sum": (x @ x).sum().item()}


if __name__ == "__main__":
    start_ray()
    try:
        if ray.cluster_resources().get("GPU", 0) < 1:
            raise SystemExit("SKIP: this cluster has no declared GPU resource.")
        result = ray.get(gpu_check.remote())
        assert result["sum"] == 4096.0
        print(result)
    finally:
        ray.shutdown()
