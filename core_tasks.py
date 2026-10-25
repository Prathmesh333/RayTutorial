import ray
from common import start_ray


@ray.remote(num_cpus=1)
def square(x):
    return x * x


@ray.remote(num_cpus=1)
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n
        return self.value


if __name__ == "__main__":
    start_ray()
    try:
        values = list(range(8))
        serial = [x * x for x in values]
        refs = [square.remote(x) for x in values]
        distributed = ray.get(refs)
        assert distributed == serial
        counter = Counter.remote()
        assert ray.get(counter.add.remote(sum(distributed))) == 140
        print({"squares": distributed, "sum": 140})
        ray.kill(counter)
    finally:
        ray.shutdown()
