import os
import ray
import numpy as np


def start_ray():
    address = os.environ.get("RAY_ADDRESS")
    if address:
        return ray.init(address=address)
    return ray.init(num_cpus=4, include_dashboard=False)


def features(batch):
    x = batch["id"].astype(np.float32) / 1023.0 * 2.0 - 1.0
    return {"x": x, "y": 3.0 * x + 2.0}


def dataset():
    return ray.data.range(1024, override_num_blocks=4).map_batches(
        features, batch_format="numpy"
    )
