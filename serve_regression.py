import os
from pathlib import Path
import torch
from ray import serve


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class Regressor:
    def __init__(self, state):
        self.model = torch.nn.Linear(1, 1)
        self.model.load_state_dict(state)
        self.model.eval()

    async def __call__(self, request):
        payload = await request.json()
        x = torch.tensor(payload["x"], dtype=torch.float32).reshape(-1, 1)
        with torch.inference_mode():
            return {"prediction": self.model(x).reshape(-1).tolist()}


# Import only a trusted checkpoint. This toy model is small enough to serialize.
state = torch.load(Path(os.environ.get("MODEL_PATH", "artifacts/model.pt")),
                   map_location="cpu", weights_only=True)
app = Regressor.bind(state)
