"""Two-GPU FSDP2 mechanics example; not a huge-model benchmark or recovery example."""
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
from pathlib import Path
import ray
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, get_device
from common import start_ray


def loop():
    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = get_device()
    torch.cuda.set_device(device)
    context = train.get_context()
    world, rank = context.get_world_size(), context.get_world_rank()
    mesh = init_device_mesh("cuda", (world,))
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 1)).to(device)
    fully_shard(model[0], mesh=mesh)
    fully_shard(model[2], mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Each rank receives a different, equally sized slice.
    x = torch.randn(128, 16, device=device).chunk(world)[rank].contiguous()
    y = x.sum(dim=1, keepdim=True)
    for step in range(4):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        train.report({"rank_local_loss": loss.item(), "step": step + 1})


if __name__ == "__main__":
    import uuid
    start_ray()
    try:
        if ray.cluster_resources().get("GPU", 0) < 2:
            raise SystemExit("SKIP: two usable GPUs are required.")
        result = TorchTrainer(
            loop, scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
            run_config=RunConfig(storage_path=str(Path("runs").resolve()),
                                 name="fsdp-" + uuid.uuid4().hex),
        ).fit()
        print(result.metrics)
    finally:
        ray.shutdown()
