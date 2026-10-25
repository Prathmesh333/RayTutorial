# Sources and claim map

Checked 25 October 2026. Dynamic documentation links showed Ray 2.59.0 when consulted. Release-tagged source anchors the APIs used where available.

| Claim | Primary source |
|---|---|
| Latest stable release, GA LLM APIs, vLLM version | https://github.com/ray-project/ray/releases/tag/ray-2.59.0 |
| Published package and platform wheels | https://pypi.org/project/ray/2.59.0/ |
| Exact latest-release metadata | https://api.github.com/repos/ray-project/ray/releases/latest |
| Ray objects and references | https://docs.ray.io/en/latest/ray-core/objects.html |
| Logical CPU/GPU resources | https://docs.ray.io/en/latest/ray-core/scheduling/resources.html |
| Node roles | https://docs.ray.io/en/latest/cluster/key-concepts.html |
| Platform support | https://docs.ray.io/en/latest/ray-overview/installation.html |
| DDP preparation and per-worker batch sizes | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/train/getting-started-pytorch.rst |
| Dataset sharding and ingest | https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html |
| Checkpoints | https://docs.ray.io/en/latest/train/user-guides/checkpoints.html |
| Train V2 recovery and deprecated restoration APIs | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/train/user-guides/fault-tolerance.rst |
| Manual cluster and network setup | https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html |
| FSDP2 mechanics | https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html |
| Tensor and pipeline parallelism | https://openai.com/index/techniques-for-training-large-neural-networks/ |
| DeepSpeed integration | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/train/deepspeed.rst |
| Ray Tune | https://docs.ray.io/en/latest/tune/getting-started.html |
| Batch inference | https://docs.ray.io/en/latest/data/batch_inference.html |
| Ray Serve | https://docs.ray.io/en/latest/serve/ |
| Exact TP, PP, placement configuration | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/serve/llm/user-guides/cross-node-parallelism.md |
| Matching cross-node code | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/serve/doc_code/cross_node_parallelism_example.py |
| vLLM engine serving | https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ |
| vLLM dependency metadata | https://pypi.org/pypi/vllm/0.27.0/json |
| LLMConfig public interface | https://github.com/ray-project/ray/blob/ray-2.59.0/python/ray/serve/llm/__init__.py |
| Observability | https://docs.ray.io/en/latest/ray-observability/getting-started.html |
| Autoscaling | https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html |
| Token authentication | https://github.com/ray-project/ray/blob/ray-2.59.0/doc/source/ray-security/token-auth.md |
| OpenAI historical use, 2022 discussion recapped 16 February 2023 | https://www.anyscale.com/blog/ray-summit-2022-stories-large-language-models |
| Uber optimization workloads, 9 January 2025 | https://www.uber.com/in/en/blog/how-uber-uses-ray-to-optimize-the-rides-business/ |
| Shopify Merlin online inference, 10 March 2023 | https://shopify.engineering/shopifys-machine-learning-platform-real-time-predictions |
| RLlib | https://docs.ray.io/en/latest/rllib/index.html |

Company examples establish the published historical workload only. They do not establish current Ray versions, unpublished model architectures, current deployment topology, or tutorial performance.

The 70-billion-parameter memory calculation is explicit arithmetic under stated assumptions. It is not a measured model footprint. The cluster diagrams and small synthetic workload are original explanatory material.
