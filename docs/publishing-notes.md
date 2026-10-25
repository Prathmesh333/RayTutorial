# Publication notes

Suggested topics: Machine Learning, Python, Distributed Systems, MLOps, Artificial Intelligence.

The article is a long tutorial with four diagrams. It intentionally distinguishes the executed CPU path from source-reviewed distributed configurations. Keep the execution-status paragraph when publishing this draft unless new tests replace it with evidence.

The complete code is hosted at https://github.com/Prathmesh333/RayTutorial and linked in the introduction and file references. Confirm that links point to the published revision before public publication.

The companion includes a CPU baseline, Core, Data, GPU scheduling, Train, Tune, batch inference, Serve, LLM configuration, and FSDP2 example. Files 03, 04, 08, and 09 have important runtime limitations documented in verification.md.

Possible follow-up articles:

1. Validate Ray Train V2 on Linux: DDP scaling, storage, and injected worker failures.
2. Train a model with FSDP2: initialization, parameter groups, activation memory, distributed checkpoints, and resume.
3. Measure Ray Serve LLM on two nodes: immutable model revision, rank placement, transport, batching, KV-cache capacity, and load tests.

Each follow-up should add measured evidence rather than repeat conceptual API snippets.
