# Ray 2.59.0 companion

This repository contains complete programs and hardware-dependent examples for [Scaling Machine Learning with Ray](docs/article.md). Read [the verification report](docs/verification.md) for actual execution status. Commands below run from the repository root. Use a dedicated development environment.

## Files

| File | Purpose | Resources |
|---|---|---|
| [local_baseline.py](local_baseline.py) | Plain PyTorch baseline; exports artifacts/model.pt | CPU |
| [core_tasks.py](core_tasks.py) | Tasks, actor, result checks | 4 logical CPUs locally |
| [data_pipeline.py](data_pipeline.py) | 1,024 rows and preprocessing | CPU |
| [gpu_check.py](gpu_check.py) | Real CUDA tensor operation | 1 usable GPU; CUDA PyTorch |
| [train_regression.py](train_regression.py) | Ray Train V2 DDP, metrics, checkpoint, export | 1 or 2 workers; Linux recommended |
| [tune_regression.py](tune_regression.py) | Two independent learning-rate trials | CPU |
| [batch_inference.py](batch_inference.py) | Model actor and checked batch predictions | CPU; exported model |
| [serve_regression.py](serve_regression.py) | Model endpoint | CPU; exported model |
| [serve_llm.py](serve_llm.py) | One TP=4, PP=2 LLM replica | Linux, 8 compatible GPUs |
| [train_fsdp.py](train_fsdp.py) | Small FSDP2 training mechanics | Linux, 2 compatible GPUs |
| [common.py](common.py) | Ray initialization and synthetic data | imported helper |

## Install and run the CPU path

Use Python 3.11. Create a virtual environment and install requirements.txt. On Windows use `.venv\Scripts\python.exe`; on Linux activate `.venv/bin/activate`. `python -m pip check` must succeed. Do not use the authoring Windows lockfile as a Linux or CUDA environment specification.

```bash
python -m pip install -r requirements.txt
python local_baseline.py
python core_tasks.py
python data_pipeline.py
python tune_regression.py
python batch_inference.py
serve run serve_regression:app
```

The baseline exports a model for batch and online inference even if distributed training is unavailable. An executed Ray Train run replaces this export with its own trained weights.

Query the local endpoint from a second terminal. On Windows PowerShell use curl.exe for this curl command, or use Invoke-RestMethod with an equivalent JSON body.

```bash
curl -X POST http://127.0.0.1:8000/ -H 'Content-Type: application/json' -d '{"x": [0, 1]}'
```

Expected: predictions close to 2 and 5. The toy function is deterministic, but small floating-point differences are normal. No timing or scalability claim is inferred.

## Ray Train path

The native Windows authoring build failed to initialize Gloo. Use Linux for the documented DDP and multi-node path. This limitation is recorded rather than bypassed with a fake process group.

```bash
python train_regression.py --workers 1 --name first-cpu-run
python train_regression.py --workers 2 --name second-cpu-run
python batch_inference.py
```

The script enables Ray Train V2 before imports. Default local runs advertise 4 logical CPUs. Supported worker counts divide 32 so every worker receives equal full batches from 1,024 rows. More workers need more physical/logical resources and appropriate input-pipeline headroom. The helper's 4-CPU local limit is a tutorial setting, not a benchmark recommendation.

All ranks call report equally often. Rank zero reports the replicated model and optimizer checkpoint. To resume an interrupted V2 run, reuse exactly the same --storage and --name. Use a different name for a fresh experiment. Do not change dataset, worker count, or algorithm during a recovery test. Epoch-boundary resume does not establish exact mid-epoch or RNG replay.

The result is a training loss, not validation accuracy. Run tune_regression.py for the separate synthetic holdout comparison.

## Join two Linux workers to one cluster

Preconditions: two worker machines with at least four available CPUs and one usable GPU each; a separate head with physical CPU/RAM capacity; identical Ray 2.59.0/Python/PyTorch CUDA environments; private connectivity; shared durable storage mounted as /mnt/shared on all participating nodes. Provision the same secret file at /secure/ray/token on every node and client, with restrictive file permissions. Set HEAD_IP and each WORKER_IP to reachable private IPv4 addresses. The path and addresses are configuration placeholders, not existing resources.

On the head:

```bash
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH=/secure/ray/token
ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
  --num-cpus=0 --num-gpus=0 --dashboard-host=127.0.0.1
```

On each worker:

```bash
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH=/secure/ray/token
ray start --address="$HEAD_IP:6379" --node-ip-address="$WORKER_IP" \
  --num-cpus=4 --num-gpus=1
```

On the head, from the companion directory:

```bash
RAY_ADDRESS=auto python train_regression.py --workers 2 --gpu \
  --storage /mnt/shared/ray-runs --name two-node-gpu-run
```

The same companion modules must be importable on every worker. This example assumes identical code installation. For remote job submission, use Ray Jobs runtime environments or an image; do not assume the driver uploads files automatically. Configure all necessary Ray and backend ports, not only 6379. Token authentication does not encrypt transport. Use a private, protected network and the official security guidance.

The head's zero logical task CPUs do not mean zero physical CPU requirements. The advertised GPU count does not manufacture GPUs. Two GPU workers in this example train replicated model copies.

## FSDP2 and large-model serving

`train_fsdp.py` requires two Linux CUDA workers with PyTorch 2.9.0. It trains a small model with FSDP2 and reports rank-local losses. It does not implement sharded checkpoint recovery and does not prove a huge model fits. For multi-node execution, configure a shared run-storage path before adapting that script.

`serve_llm.py` uses a separate Linux GPU environment:

```bash
python -m pip install -r requirements-llm.txt
python -m pip check
RAY_ADDRESS=auto serve run serve_llm:app
```

Do not install the training requirements in this environment. Ray's llm extra selects vLLM 0.27.0; vLLM selects its own Torch/CUDA stack. The authoring environment could not install or execute this Linux GPU stack, so the file is a version-matched configuration candidate, not a tested lockfile. Follow the vLLM platform and driver requirements and inspect the resolved environment before running.

Plan for two four-GPU machines with CPU headroom for engine workers and serving processes; 16 available CPU cores per machine is an illustrative configuration. Eight 80-GB-class NVIDIA GPUs are an illustrative target, not a tested minimum or a guarantee. This is a different resource profile from the two-worker training cluster above. The model downloads substantial data and needs host memory and local disk. Check model permissions and license. All workers need a usable model source. Qwen's model identifier is fixed here, but model repository revisions are not locked; select an immutable model revision for a reproducible production deployment.

One replica uses 4 tensor-parallel ranks x 2 pipeline stages = 8 GPU workers. PACK is a preference; verify rank placement. Serving uses bfloat16 and a 4096-token context setting. Verify CUDA, engine, architecture, collectives, KV-cache capacity, memory peaks, and actual requests. There is no measured fit or latency claim.

Example request after a successful deployment:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-72b","messages":[{"role":"user","content":"Explain data parallelism briefly."}],"max_tokens":64}'
```

This endpoint shape is OpenAI-compatible; it does not call OpenAI. This request was not executed on the authoring machine. This is a local/private demonstration, not an authenticated public API configuration.

## Cleanup

Stop the foreground Serve command with Ctrl+C. Use `serve shutdown` to remove applications from the selected tutorial cluster. Run `ray stop` on each dedicated tutorial node after all jobs finish. Do not run these commands on a shared cluster containing unrelated jobs. Python ray.shutdown disconnects from externally managed clusters. External checkpoints and paid machines need separate lifecycle management; none are provisioned by these examples.

## Diagram sources

Each PNG has an editable SVG sibling in diagrams. tools/make_diagrams.py reproduces the four architecture diagrams. tools/make_tables.py reproduces the memory and decision tables for Medium. The GitHub article retains searchable Markdown tables. Rendering needs Pillow; the authoring renderer used Pillow 11.0.0. Adjust FONT if your system uses another font location. This dependency is separate from Ray execution.
