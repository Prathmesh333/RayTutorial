# Scaling Machine Learning with Ray: From One Machine to Distributed Training and Serving

Train a small model, distribute its work, and choose the right approach when one GPU is no longer enough.

Adding another graphics processing unit (GPU) can speed up training. It can also leave you with the same out-of-memory error. The result depends on what you distribute: data, model state, or inference requests.

We will start with a small Python model, then follow its data through training and serving. The final sections explain what changes when a model cannot fit on one GPU or one machine.

Examples use **Ray 2.59.0 and Python 3.11**. [Ray release](https://github.com/ray-project/ray/releases/tag/ray-2.59.0).

## 1. Train a small model and see a prediction

The example learns `y = 3x + 2` from 1,024 synthetic rows. For an input of zero, the target is two. For an input of one, the target is five. This simple relationship lets us check each stage without downloading a model or dataset.

Clone [RayTutorial](https://github.com/Prathmesh333/RayTutorial), then create a virtual environment:

```bash
git clone https://github.com/Prathmesh333/RayTutorial.git
cd RayTutorial
python -m venv .venv
```

Activate the environment. On Linux or macOS, run:

```bash
source .venv/bin/activate
```

On Windows PowerShell, run this command instead:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies, check them, and run the baseline:

```bash
python -m pip install -r requirements.txt
python -m pip check
python local_baseline.py
```

[local_baseline.py](https://github.com/Prathmesh333/RayTutorial/blob/main/local_baseline.py) trains a PyTorch linear model without Ray. It saves weights to `artifacts/model.pt` and prints the loss, learned coefficients, and predictions.

The prediction check produces values close to:

```text
Input:       [0, 1]
Prediction:  [2, 5]
```

The script prints full-precision values. The values above are rounded for readability. The later batch and online inference examples load these trained weights.

The local examples use four available central processing unit (CPU) cores and sufficient memory for Python, PyTorch, and Ray processes. [requirements.txt](https://github.com/Prathmesh333/RayTutorial/blob/main/requirements.txt) pins the CPU environment's direct dependencies. The [README](https://github.com/Prathmesh333/RayTutorial#readme) provides the complete file guide and commands. Each linked Python file is a separate program; run commands from the repository root.

## 2. Run your first distributed Python task

Before distributing model training, distribute a small function. The following standalone example contains the task pattern used in [core_tasks.py](https://github.com/Prathmesh333/RayTutorial/blob/main/core_tasks.py):

```python
import ray
@ray.remote(num_cpus=1)
def square(x):
    return x * x
if __name__ == "__main__":
    ray.init(num_cpus=4, include_dashboard=False)
    try:
        refs = [square.remote(x) for x in range(8)]
        print(ray.get(refs))
    finally:
        ray.shutdown()
```

`square.remote()` submits a task and returns an object reference, or `ObjectRef`. It does not wait for the result. The list therefore submits all eight tasks before `ray.get()` retrieves their results.

Expected output:

```text
[0, 1, 4, 9, 16, 25, 36, 49]
```

Calling `ray.get()` immediately after each submission can serialize the loop. This tiny calculation demonstrates scheduling; task overhead can make it slower than ordinary Python. [Ray task patterns](https://docs.ray.io/en/latest/ray-core/patterns/ray-get-loop.html).

Run `python core_tasks.py` to check both tasks and an actor. A task executes a remote function. An actor keeps state inside a remote Python object. The example's actor accumulates a checked total of 140.

Resource declarations control scheduling. `num_cpus=1` requests one logical CPU resource; it does not pin a process to a physical core or constrain every library thread. Similarly, `num_gpus=1` requests a GPU resource. It does not partition GPU memory or move tensors automatically. [Ray resource semantics](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html).

## 3. Understand what changes when another machine joins

Ray schedules tasks and actors across one cluster. A driver submits the application. The head node runs cluster services, including the Global Control Service (GCS). Worker nodes contribute compute resources. Each node has a raylet that participates in scheduling and resource management. [Ray cluster concepts](https://docs.ray.io/en/latest/cluster/key-concepts.html).

![A driver submits work to one Ray cluster. The head coordinates CPU and GPU worker nodes. Worker nodes have raylets and object stores, and transfer objects between nodes.](../diagrams/01-cluster.png)

*One cluster can contain CPU and GPU machines. Model frameworks handle tensor communication between training or inference workers.*

Ray uses per-node object stores to share supported objects and transfer data between nodes. An `ObjectRef` identifies a result that can become available later. The object store does not combine GPU memory into one allocation. [Ray objects](https://docs.ray.io/en/latest/ray-core/objects.html).

The division of responsibility matters: Ray manages workers and resources. PyTorch manages the neural network, automatic differentiation, and distributed training operations. An inference engine such as vLLM manages model execution and engine-level batching. Adding a machine does not automatically partition an arbitrary model.

## 4. Feed data to two training workers

[data_pipeline.py](https://github.com/Prathmesh333/RayTutorial/blob/main/data_pipeline.py) creates the same 1,024-row problem with Ray Data. Run `python data_pipeline.py` to check the row count and target relationship.

The transformation in [common.py](https://github.com/Prathmesh333/RayTutorial/blob/main/common.py) converts identifiers into features and targets:

```python
import numpy as np
import ray
def features(batch):
    x = batch["id"].astype(np.float32) / 1023.0 * 2.0 - 1.0
    return {"x": x, "y": 3.0 * x + 2.0}
def dataset():
    return ray.data.range(1024, override_num_blocks=4).map_batches(
        features, batch_format="numpy"
    )
```

For real data, a reader such as `ray.data.read_parquet()` replaces the synthetic source. Every node that performs reads needs access to the data location. A local path on one machine is not automatically available on another. Ray Data supplies dataset shards to training workers. [Data loading and training integration](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html).

![Ray Data preprocesses synthetic rows. Ray Train supplies shards to PyTorch workers. Training saves a checkpoint, and the resulting weights feed batch inference or an HTTP service.](../diagrams/02-workflow.png)

*The same small model connects preprocessing, training, checkpoint storage, batch inference, and online serving.*

[train_regression.py](https://github.com/Prathmesh333/RayTutorial/blob/main/train_regression.py) connects those parts. The following excerpt is its trainer configuration; the complete file defines `train_loop`, `dataset`, and the parsed command-line arguments:

```python
trainer = TorchTrainer(
    train_loop,
    train_loop_config={"epochs": args.epochs, "lr": args.lr},
    scaling_config=ScalingConfig(
        num_workers=args.workers,
        use_gpu=args.gpu,
        resources_per_worker={"CPU": 1},
        placement_strategy="SPREAD",
    ),
    datasets={"train": dataset()},
    run_config=RunConfig(
        storage_path=args.storage,
        name=args.name,
        failure_config=FailureConfig(max_failures=1),
    ),
)
result = trainer.fit()
```

`TorchTrainer` launches the worker group. `ScalingConfig` requests workers and their resources. `datasets` supplies training data. `RunConfig` identifies the output location and recovery policy. `fit()` starts the run.

Inside `train_loop`, `prepare_model()` places `torch.nn.Linear(1, 1)` on the assigned device. With multiple workers, it enables DistributedDataParallel (DDP). Each worker reads different data while PyTorch synchronizes gradients across model replicas. [Ray PyTorch guide, release-tagged source](https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/train/getting-started-pytorch.rst).

Use Linux for the distributed training examples below.

Start with one worker. Use a new run name when comparing two workers:

```bash
python train_regression.py --workers 1 --name cpu-one-worker
python train_regression.py --workers 2 --name cpu-two-workers
```

The worker batch size is 32. Two workers produce an effective global batch size of 64 without gradient accumulation. Increasing workers therefore changes learning behavior unless another setting compensates.

The script restricts worker counts so that every worker receives equal full batches. Compatible collective operations need compatible iteration counts; otherwise, one worker can wait indefinitely for another. The reported loss is an example-weighted training loss across workers. After training, the driver exports `artifacts/model.pt` for inference.

## 5. Save the state needed to resume

A checkpoint must contain the state that the training loop needs after a restart. This example saves the model, optimizer, and next epoch. Every worker reports once per epoch. Only rank zero writes the replicated DDP checkpoint. A sharded model needs a different checkpoint strategy. [Ray checkpoint guide](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html).

The script enables `RAY_TRAIN_V2_ENABLED=1` before importing Ray Train. Its configuration allows one worker-group recovery attempt. The loop calls `ray.train.get_checkpoint()` and restores state when Ray supplies a checkpoint.

For driver recovery, reuse the same storage path and run name. Use a new name for a new experiment. Ray Train V2 uses that identity to locate persisted run state. The example avoids deprecated `TorchTrainer.restore()` calls. [Ray Train recovery](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html).

Recovery is designed around epoch boundaries. It does not promise identical mid-epoch replay or restore every random-number-generator state. Multi-node runs also need durable storage accessible to all participating nodes. Identical path names on independent disks do not create shared storage.

## 6. Move training from one GPU to two machines

First check the GPU environment with [gpu_check.py](https://github.com/Prathmesh333/RayTutorial/blob/main/gpu_check.py). It performs a real CUDA tensor operation. The script skips if Ray advertises no GPU; an advertised GPU can still fail if the driver or PyTorch build is incompatible.

```bash
python gpu_check.py
```

GPU training requires Linux, a compatible CUDA-enabled PyTorch build, NVIDIA drivers, and suitable hardware. Use the [Ray installation guide](https://docs.ray.io/en/latest/ray-overview/installation.html) to prepare the environment.

On a Linux machine with two usable GPUs, this command requests one GPU per training worker:

```bash
python train_regression.py --workers 2 --gpu --name gpu-two-workers
```

For two Linux machines, join worker nodes to the same Ray cluster. The [README's cluster setup](https://github.com/Prathmesh333/RayTutorial#join-two-linux-workers-to-one-cluster) provides the commands and token configuration. The example uses a separate head and two workers, each with four available CPUs and one GPU.

The head advertises zero logical task CPUs while retaining physical CPUs for cluster services. Nodes need compatible environments, reachable addresses, and all ports required by Ray and the distributed backend. Opening only port 6379 is insufficient. [Manual cluster setup](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).

Once the authenticated cluster and shared storage exist, launch the driver on the head:

```bash
RAY_ADDRESS=auto python train_regression.py \
  --workers 2 --gpu \
  --storage /mnt/shared/ray-runs \
  --name gpu-two-nodes
```

`SPREAD` prefers distributing workers when resources permit. It does not guarantee a specific GPU-to-rank topology. DDP still gives each worker a complete model copy.

Separate Ray clusters do not automatically form one training job. Independent jobs need explicit orchestration and data or checkpoint exchange. Cross-region synchronous training also faces network latency, bandwidth limits, and additional failure concerns. These examples use one cluster in one low-latency network.

## 7. Understand why another GPU may not solve memory limits

Data-parallel workers process more data concurrently, but each worker still needs room for its model replica. To reduce replicated model state, choose an explicit sharding strategy.

![Two DDP workers each contain a complete model and different data batches. Two fully sharded workers hold different persistent state shards and communicate during computation.](../diagrams/03-training.png)

*DDP increases parallel data processing. Full sharding reduces replicated persistent model state, but computation still needs temporary memory.*

**Fully Sharded Data Parallel (FSDP)** partitions model state across workers. PyTorch FSDP2 gathers parameter groups when computation needs them. Its `fully_shard()` API applies sharding before optimizer creation. [PyTorch 2.9 FSDP2](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html).

[train_fsdp.py](https://github.com/Prathmesh333/RayTutorial/blob/main/train_fsdp.py) demonstrates that mechanism with a small model and two GPUs. Ray Train establishes the workers; PyTorch creates the device mesh and shards the model. It does not wrap the sharded model in DDP. The example does not implement sharded checkpoint recovery.

**Tensor parallelism** splits tensor operations within layers. Frequent communication makes the interconnect important. **Pipeline parallelism** assigns sequences of layers to different workers. Activations move between stages; stage imbalance and idle pipeline periods can limit throughput. [Large-model training techniques](https://openai.com/index/techniques-for-training-large-neural-networks/).

DeepSpeed is another option inside a Ray Train worker group. The training framework determines the sharding strategy while Ray provides orchestration. [Ray DeepSpeed integration](https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/train/deepspeed.rst).

Consider a hypothetical model with 70 billion parameters. The following estimate assumes two-byte weights and gradients, two 32-bit floating-point (FP32) Adam moment buffers, and an optional FP32 master-weight copy:

| Stored state | Assumption | Approximate memory |
|---|---|---|
| Weights | 2 bytes per parameter | 140 GB |
| Gradients | 2 bytes per parameter | 140 GB |
| Adam moment buffers | Two FP32 buffers | 560 GB |
| Master-weight copy, if used | 4 bytes per parameter | 280 GB |
| Total with master weights | 16 bytes per parameter | 1.12 TB |

These are decimal GB and TB. The estimate excludes activations, temporary allocations, fragmentation, and communication buffers. Precision and optimizer implementations change the total. Sharding reduces persistent state per worker but does not eliminate peak parameter gathers or activation memory. Dividing 1.12 TB by a GPU count is not a sufficient capacity test.

## 8. Compare learning rates without confusing trials with workers

Run [tune_regression.py](https://github.com/Prathmesh333/RayTutorial/blob/main/tune_regression.py) to compare learning rates of 0.01 and 0.1:

```bash
python tune_regression.py
```

Each trial trains a separate model and reports loss on separate synthetic evaluation inputs. Ray Tune selects the lower validation loss. The example selects 0.1 for this synthetic problem. Use a representative validation dataset when tuning a real model. [Ray Tune guide](https://docs.ray.io/en/latest/tune/getting-started.html).

Two trials produce two models. Two DDP workers cooperate on one model. Combining both dimensions multiplies resource requirements: four concurrent trials with two GPUs each require eight GPUs before other workloads are considered.

## 9. Use the trained weights for batch and online predictions

[batch_inference.py](https://github.com/Prathmesh333/RayTutorial/blob/main/batch_inference.py) loads the exported weights in a Ray Data actor. The actor loads its model once and predicts batches:

```bash
python batch_inference.py
```

The script checks 1,024 outputs and their mean squared error against the known function. These tiny weights can be serialized from the driver. Large models should load from shared artifact storage instead. [Ray Data batch inference](https://docs.ray.io/en/latest/data/batch_inference.html).

For online predictions over Hypertext Transfer Protocol (HTTP), [serve_regression.py](https://github.com/Prathmesh333/RayTutorial/blob/main/serve_regression.py) defines a Ray Serve deployment. Run this command from the repository root:

```bash
serve run serve_regression:app
```

Send a request from another Bash terminal:

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H 'Content-Type: application/json' \
  -d '{"x": [0.0, 1.0]}'
```

The endpoint returns predictions close to 2 and 5 with the trained baseline weights. You have now followed the same small model from training to a request-response service. The local example omits production input validation and authentication. [Ray Serve](https://docs.ray.io/en/latest/serve/).

Additional independent replicas can increase request capacity when resources permit. Each replica still needs memory for its complete model. Replicas do not combine their memory to execute one oversized model.

## 10. Serve one model across two machines

When one model exceeds a machine's available memory, its inference workers must cooperate across devices. Ray Serve LLM supports tensor and pipeline parallelism through vLLM. [Ray Serve LLM cross-node guide](https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/serve/llm/user-guides/cross-node-parallelism.md).

![Independent serving replicas each contain a full model. A partitioned replica spans two four-GPU nodes, with tensor parallelism within each pipeline stage and activations exchanged between stages.](../diagrams/04-serving.png)

*A complete-model replica serves requests independently. A partitioned replica needs its participating workers for each distributed inference operation.*

[serve_llm.py](https://github.com/Prathmesh333/RayTutorial/blob/main/serve_llm.py) targets `Qwen/Qwen2.5-72B-Instruct`. One logical replica uses four tensor-parallel workers across two pipeline stages: eight GPU workers in total. The fixed replica count prevents an automatic request for another eight-GPU replica.

This excerpt shows the engine settings. The linked file contains the full application configuration:

```python
engine_kwargs={
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 2,
    "distributed_executor_backend": "ray",
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
}
```

On Linux, create a separate [requirements-llm.txt](https://github.com/Prathmesh333/RayTutorial/blob/main/requirements-llm.txt) environment. It pins Ray 2.59.0 and vLLM 0.27.0. Do not combine it with the tutorial's PyTorch 2.9.0 training environment; vLLM supplies its own PyTorch and CUDA requirements.

After preparing that environment and a compatible cluster, launch the application:

```bash
serve run serve_llm:app
```

On two nodes with exactly four available GPUs each, the eight workers require both nodes. `PACK` prefers fewer nodes but does not guarantee rank-to-node assignment. Confirm placement in engine logs. The diagram shows the intended arrangement, not guaranteed placement.

Every participating machine needs access to the model repository. Downloads, host memory, disk space, GPU memory, and inter-node communication affect startup. Weights are only part of inference memory. The key-value (KV) cache stores attention state; engine workspaces and communication buffers consume additional memory.

The 4,096-token context setting and 85-percent memory-utilization setting do not prove that this model fits. Ray Serve routes requests to the logical deployment. vLLM executes the model across its workers and manages engine-level batching. Another independent replica requires another full set of partition workers.

Slow links can make cross-node inference unsuitable for a latency-sensitive service. Measure time to first token, output-token rate, tail latency, and memory under the intended concurrency. [vLLM distributed serving](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).

## 11. Diagnose the bottleneck before adding resources

Use `ray status` and the Ray dashboard to distinguish pending work from running work. A pending task can indicate an unavailable resource bundle. Idle GPUs can indicate slow data reads or insufficient CPU preprocessing. High object-store usage can indicate too many retained object references. [Ray observability](https://docs.ray.io/en/latest/ray-observability/getting-started.html).

Other limits need different responses. Small tasks can spend more time scheduling than computing. Large batches can exhaust activation memory even when weights fit. Checkpoints reduce recoverable work loss after a failure, but replacement capacity and storage bandwidth still affect recovery time.

Autoscaling needs a configured infrastructure provider or cluster operator. `num_workers=8` requests workers; it does not independently purchase machines. Replica autoscaling and cluster-node autoscaling are separate controls. For Kubernetes, evaluate KubeRay and its resource configuration. [Ray autoscaling](https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html).

Ray 2.59 enables token authentication by default for newly initialized local clusters. Manual remote clusters need explicit token configuration. Configure the same secret on nodes and clients, protect the network, and keep management endpoints off the public internet. Tokens alone do not encrypt traffic. [Ray token authentication](https://docs.ray.io/en/latest/ray-security/token-auth.html).

Include storage, data transfer, checkpoint traffic, idle replicas, and recovery time in cost estimates. When finished, stop the foreground Serve command. Use `serve shutdown` for the selected development cluster. Run `ray stop` on each dedicated tutorial node after all jobs finish.

The Python scripts call `ray.shutdown()` to disconnect; that call does not tear down a separately started cluster. These cleanup commands do not delete external checkpoints or terminate cloud machines. Keep shared-cluster workloads separate from tutorial cleanup.

## 12. See where companies have used Ray

These company examples show how Ray supports different workloads.

**OpenAI:** Anyscale's 16 February 2023 recap links Greg Brockman's Ray Summit 2022 discussion of Ray in OpenAI's distributed platform and large-model training. [Conference recap and speaker clips](https://www.anyscale.com/blog/ray-summit-2022-stories-large-language-models).

**Uber:** its 9 January 2025 account combines Spark data processing with Ray execution of parallel Python optimization tasks for the rides marketplace. Ray's role here extends beyond neural-network training. [Uber engineering](https://www.uber.com/in/en/blog/how-uber-uses-ray-to-optimize-the-rides-business/).

**Shopify:** its 10 March 2023 account describes adding online inference to the Ray-based Merlin platform. [Shopify engineering](https://shopify.engineering/shopifys-machine-learning-platform-real-time-predictions).

Other applications include simulation, reinforcement learning with RLlib, feature processing, model evaluation, embeddings, and independent experiments. The shared requirements are explicit units of work, resource declarations, and controlled data movement. [Ray use cases](https://docs.ray.io/en/latest/ray-overview/use-cases.html), [RLlib](https://docs.ray.io/en/latest/rllib/index.html).

## Choose the next change from the problem you observe

| Observed problem | Approach to evaluate | Condition to check |
|---|---|---|
| Training is slow, but the model fits | Data-parallel workers | Input throughput and communication cost |
| Training state exceeds device memory | FSDP or another sharding strategy | Peak gathers, activations, and checkpoints |
| Inference requests exceed one replica's capacity | More independent replicas | A complete model fits each replica |
| One inference model exceeds device or machine memory | Tensor or pipeline parallelism | Engine support, placement, and interconnect |
| Independent experiments take too long | Concurrent Tune trials | Resources for every concurrent trial |

Begin with the CPU baseline and confirm its predictions. Then identify whether your next constraint is throughput, memory, or request capacity. That constraint determines what to distribute before it determines how many GPUs to request.
