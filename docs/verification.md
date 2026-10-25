# Verification report

Date: 25 October 2026. This report distinguishes execution evidence from source review.

## Environment and version

CPython 3.11.3 on Windows. Ray 2.59.0. PyTorch 2.9.0+cpu. Ray detects a local NVIDIA GPU, but this PyTorch wheel has no CUDA support. No usable CUDA execution environment or second machine was available. The existing Ubuntu WSL registration could not start because its virtual disk path was missing; no system repair or installation was attempted.

GitHub's latest-release API returned ray-2.59.0, published 2 October 2026, prerelease=false. PyPI returned 2.59.0. The article gives the pinned version; this report records the verification date. The pinned CPU environment passed `pip check`. The complete authoring dependency snapshot is included separately.

## Executed successfully

| Check | Evidence | Scope |
|---|---|---|
| Plain PyTorch training, local_baseline.py | MSE 7.91001986133466e-12; weight 2.99999737739563; bias 2.000002384185791 | Synthetic in-sample regression check only |
| Ray Core tasks and actor, core_tasks.py | Squares [0,1,4,9,16,25,36,49]; sum 140; process exited zero | Local remote tasks and persistent actor state |
| Ray Data, data_pipeline.py | 1,024 rows; target relation checked for every row; exit zero | Actual dataset execution |
| Ray Tune, tune_regression.py | Both trials completed; selected lr=0.1; validation loss 4.441388591658324e-05 | Separate synthetic evaluation inputs, not real model quality |
| Batch inference, batch_inference.py | 1,024 outputs; MSE 7.980009014696421e-12; exit zero | Weights came from the plain PyTorch baseline, not a successful Ray Train run |
| Online serving, serve_regression.py | HTTP 200; predictions [2.000002384185791,5.0]; tolerance check passed | Actual Ray Serve deployment and HTTP request on loopback port 18080 |
| Dependency consistency | pip check: no broken requirements | CPU environment only |
| Python syntax | All companion Python files compile | Syntax evidence only; not a distributed execution claim |

The Serve test imported the same app as the documented CLI and called serve.start/serve.run programmatically. It used port 18080 to avoid the default port. It shut down Serve and disconnected Ray afterward. Log timings are incidental observations, not benchmarks.

## Attempted but blocked

`train_regression.py --name cpu-smoke-01` reached Ray Train V2 worker setup and failed in PyTorch's Gloo process-group initialization with `makeDeviceForHostname(): unsupported gloo device`. Windows socket warnings also occurred. No successful distributed training step, checkpoint save, or checkpoint restore was observed. Do not interpret the failed attempt as a successful test of the loop.

The failure was preserved in the local authoring log. Public validation files include compact exit-status summaries; machine-specific raw logs are not published. PyTorch's upstream issue tracker documents the same error family at https://github.com/pytorch/pytorch/issues/150381. This report does not claim that a specific workaround fixes this installed wheel. The tutorial routes distributed examples to Linux and labels their execution status explicitly.

The initial dependency installation also exposed an incompatible protobuf downgrade. The final requirements use protobuf 7.36.2 and pass pip check. Only the final environment is the dependency snapshot.

## Reviewed against versioned sources, not executed

- Ray Train V2 dataset sharding, DDP preparation, metric reporting, rank-zero replicated checkpoints, and recovery identity. Interfaces matched the installed 2.59.0 source and tagged documentation. The blocked run prevents runtime confirmation.
- CUDA scheduling and tensor operation in gpu_check.py. A declared Ray GPU resource does not make the CPU-only Torch wheel CUDA-capable.
- Two-worker GPU DDP and two-node cluster commands. No second machine or compatible CUDA runtime was used.
- FSDP2 in train_fsdp.py. Reviewed against PyTorch 2.9 fully_shard documentation. No GPU execution or sharded checkpoint recovery was tested. The script is a small mechanics demonstration.
- Eight-GPU TP=4, PP=2 serving configuration in serve_llm.py. LLMConfig fields and builder shape match the Ray 2.59.0 source. Ray's package metadata selects vLLM 0.27.0. The separate Linux GPU environment was not resolved, installed, or executed. No model download, memory fit, inference response, or latency was verified.
- Model repository revision is not immutable in the example. Select a model revision before a reproducibility-sensitive deployment.
- Cloud storage, remote token distribution, cluster autoscaling, failure injection, and cross-region behavior were not exercised.

## Numerical and diagram review

- Worker batch size 32 x 2 workers = global batch size 64, with no gradient accumulation.
- 4 concurrent trials x 2 GPUs = 8 GPUs, excluding other workloads.
- TP=4 x PP=2 = 8 GPU workers per logical replica. PACK does not establish rank-to-node topology.
- Hypothetical 70B model: weights 140 GB + gradients 140 GB + FP32 Adam moments 560 GB + optional FP32 master weights 280 GB = 1,120 GB = 1.12 decimal TB. This is an assumption-based state estimate, not a measured footprint. Activations and transient allocations are excluded explicitly.
- Four original diagrams have PNG and editable SVG versions. The serving diagram illustrates an intended two-node topology and explicitly warns that placement must match. The workflow diagram represents the design; the successful inference tests used the local baseline export.

## Language and technical-equivalence review

Writing skill: controlled-technical-english. Document types: technical descriptions and operational commands. Baseline house rules used. Reader assumption: Python developers with basic ML knowledge. Project terminology comes from Ray/PyTorch sources. American spelling is a project choice. No external-standard compliance is claimed.

Reviewed subjects, actors, resource declarations, data versus model parallelism, checkpoint sequence, recovery conditions, negation, and modal force. Abbreviations are introduced in the text or used as official product names. Descriptive sentences were screened against the 24-word target; longer numerical explanations retain their qualifiers. Paragraphs and long clauses were reviewed for ambiguity rather than shortened mechanically.

Preserved numbers, units, class names, paths, example identities, and conditions. Distinguishes requirements from recommendations and possibilities. GPU memory estimates and company usage remain explicitly qualified. Safety-classification rules are not applicable; no hazard labels were invented.

## Sources and link checks

sources.md maps claims to primary documents. Some official web pages returned rate limits through the web reader; the corresponding Ray 2.59.0 tagged sources were read directly. A direct Python link probe encountered certificate validation failures for several documentation domains, plus 403/406 responses for some company pages. No certificate verification was disabled. Those probe failures do not establish broken links. The web research tool retrieved the principal Ray, PyTorch, Uber, Shopify, and conference sources. validation/links.json records the probe result without claiming universal link availability.

## Publication limits

The Medium draft remains unpublished. Complete companion files are hosted in this repository and linked from the article. GPU configuration validation, Linux DDP execution, and recovery testing remain necessary before presenting those paths as tested recipes.


## Readability revision

Applied the controlled-technical-english skill baseline to the mixed tutorial and descriptive text. Kept project API names, numerical assumptions, citations, prerequisites, recovery limits, and unexecuted GPU status. No external controlled-language certification is claimed.

The article now begins with the baseline prediction, introduces each file at its point of use, explains the trainer excerpt, and compares scaling choices in tables. The full file guide remains in the README. Medium uses image versions of the two tables and explicit Python, Bash, or Plain Text code languages.

The baseline now prints predictions for inputs 0 and 1. A fresh CPU run returned [2.000002384185791, 5.0], with MSE 7.91001986133466e-12. The additional print statement does not change training. Distributed training, GPU, recovery, and large-model execution status remain unchanged.
