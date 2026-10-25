"""Linux GPU example; source-reviewed, not executed on the authoring machine."""
from ray.serve.llm import LLMConfig, build_openai_app


config = LLMConfig(
    model_loading_config={"model_id": "qwen-72b", "model_source": "Qwen/Qwen2.5-72B-Instruct"},
    deployment_config={"num_replicas": 1},
    engine_kwargs={
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "distributed_executor_backend": "ray",
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
    },
    placement_group_config={
        "bundle_per_worker": {"CPU": 1, "GPU": 1},
        "strategy": "PACK",
    },
)
app = build_openai_app({"llm_configs": [config]})
