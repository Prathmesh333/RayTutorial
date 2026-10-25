import ray
from common import dataset, start_ray


if __name__ == "__main__":
    start_ray()
    try:
        ds = dataset()
        rows = ds.take_all()
        assert len(rows) == 1024
        assert all(abs(r["y"] - (3 * r["x"] + 2)) < 1e-5 for r in rows)
        print({"rows": len(rows), "sample": rows[:2]})
    finally:
        ray.shutdown()
