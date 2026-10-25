import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import argparse
from pathlib import Path
import tempfile
import ray
import ray.train as train
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_model
import torch
import torch.distributed as dist
from common import dataset, start_ray


def train_loop(config):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    ctx = train.get_context()
    rank = ctx.get_world_rank()
    device = get_device()
    model = prepare_model(torch.nn.Linear(1, 1))
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    start_epoch = 0
    previous = train.get_checkpoint()
    if previous:
        with previous.as_directory() as path:
            state = torch.load(Path(path) / "state.pt", map_location=device, weights_only=True)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["next_epoch"]
    shard = train.get_dataset_shard("train")
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        totals = torch.zeros(2, device=device)
        for batch in shard.iter_torch_batches(batch_size=32, dtypes=torch.float32, device=device):
            x, y = batch["x"].reshape(-1, 1), batch["y"].reshape(-1, 1)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            totals[0] += loss.detach() * x.shape[0]
            totals[1] += x.shape[0]
        if ctx.get_world_size() > 1:
            dist.all_reduce(totals)
        metrics = {"loss": (totals[0] / totals[1]).item(), "epoch": epoch + 1}
        # All ranks report once per epoch; only rank zero writes this DDP checkpoint.
        with tempfile.TemporaryDirectory() as path:
            checkpoint = None
            if rank == 0:
                state = {"model": raw_model.state_dict(),
                         "optimizer": optimizer.state_dict(), "next_epoch": epoch + 1}
                torch.save(state, Path(path) / "state.pt")
                torch.save(raw_model.state_dict(), Path(path) / "model.pt")
                checkpoint = Checkpoint.from_directory(path)
            train.report(metrics, checkpoint=checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--storage", default=str(Path("runs").resolve()))
    parser.add_argument("--name", required=True, help="Unique for a new run; reuse only to resume.")
    args = parser.parse_args()
    if args.workers < 1 or 1024 % (32 * args.workers):
        parser.error("workers must divide 32 so each shard has the same number of full batches")
    start_ray()
    try:
        trainer = TorchTrainer(
            train_loop,
            train_loop_config={"epochs": args.epochs, "lr": args.lr},
            scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=args.gpu,
                                         resources_per_worker={"CPU": 1}, placement_strategy="SPREAD"),
            datasets={"train": dataset()},
            run_config=RunConfig(storage_path=args.storage, name=args.name,
                                 failure_config=FailureConfig(max_failures=1)),
        )
        result = trainer.fit()
        if result.checkpoint is None:
            raise RuntimeError("No checkpoint was produced.")
        with result.checkpoint.as_directory() as path:
            state = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True)
        target = Path("artifacts")
        target.mkdir(exist_ok=True)
        torch.save(state, target / "model.pt")
        print({"metrics": result.metrics, "export": str(target / "model.pt")})
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
