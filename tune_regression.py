import tempfile
from pathlib import Path
import ray
from ray import tune
from ray.tune import RunConfig
import torch
from common import start_ray


def trial(config):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.linspace(-1, 1, 1024).reshape(-1, 1)
    y = 3 * x + 2
    # Separate, unseen evaluation inputs for selecting a learning rate.
    vx = torch.linspace(-0.99, 0.99, 257).reshape(-1, 1)
    vy = 3 * vx + 2
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    for _ in range(80):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    with torch.inference_mode():
        loss = torch.nn.functional.mse_loss(model(vx), vy).item()
    tune.report({"validation_loss": loss})


if __name__ == "__main__":
    start_ray()
    try:
        with tempfile.TemporaryDirectory() as output:
            results = tune.Tuner(
                tune.with_resources(trial, {"cpu": 1}),
                param_space={"lr": tune.grid_search([0.01, 0.1])},
                tune_config=tune.TuneConfig(metric="validation_loss", mode="min", max_concurrent_trials=2),
                run_config=RunConfig(storage_path=str(Path(output).resolve()), name="learning-rate"),
            ).fit()
            best = results.get_best_result()
            assert not any(result.error for result in results)
            print({"best_config": best.config, "validation_loss": best.metrics["validation_loss"]})
    finally:
        ray.shutdown()
