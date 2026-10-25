import argparse
from pathlib import Path
import ray
import numpy as np
from common import dataset, start_ray


class Predictor:
    def __init__(self, state):
        import torch
        self.torch = torch
        self.model = torch.nn.Linear(1, 1)
        self.model.load_state_dict(state)
        self.model.eval()

    def __call__(self, batch):
        x = self.torch.as_tensor(np.array(batch["x"], copy=True), dtype=self.torch.float32).reshape(-1, 1)
        with self.torch.inference_mode():
            result = self.model(x).numpy().reshape(-1)
        return {"x": batch["x"], "prediction": result}


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/model.pt")
    args = parser.parse_args()
    # Tiny trusted weights are serialized to actors. Large weights need shared model storage.
    state = torch.load(Path(args.model), map_location="cpu", weights_only=True)
    start_ray()
    try:
        predictions = dataset().map_batches(Predictor, fn_constructor_args=(state,),
                                            batch_format="numpy", batch_size=64,
                                            compute=ray.data.ActorPoolStrategy(size=1), num_cpus=1)
        rows = predictions.take_all()
        assert len(rows) == 1024
        mse = sum((r["prediction"] - (3 * r["x"] + 2)) ** 2 for r in rows) / len(rows)
        print({"rows": len(rows), "mse": mse, "sample": rows[:2]})
        assert mse < 0.05
    finally:
        ray.shutdown()
