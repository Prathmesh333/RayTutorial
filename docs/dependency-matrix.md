# Dependency and API matrix

Verified 25 October 2026. Latest stable Ray was checked through the GitHub latest-release API and PyPI JSON metadata. Both returned 2.59.0. GitHub marked the release non-prerelease and published it on 2 October 2026.

| Component | Pin or scope | Verification |
|---|---|---|
| Python | CPython 3.11.3 authoring environment | Executed on Windows |
| Ray | 2.59.0 | Installed; Core, Data, Tune, Serve executed |
| PyTorch training | 2.9.0; installed build 2.9.0+cpu | Plain model executed; Gloo process-group initialization failed |
| NumPy | 2.2.6 | Executed |
| pandas | 2.3.3 | Installed; Ray Data environment |
| PyArrow | 21.0.0 | Installed; Ray Data executed |
| protobuf | 7.36.2 | Installed; pip check passed |
| vLLM | 0.27.0, separate Linux GPU environment | Matches Ray 2.59 package metadata; not installed or executed |
| PyTorch inference | selected by vLLM 0.27.0 | PyPI metadata specifies torch==2.13.0; stack not resolved or tested here |
| CUDA, NCCL, driver | hardware- and wheel-dependent | No compatible GPU runtime test performed; not a validated matrix |
| Model | Qwen/Qwen2.5-72B-Instruct | Configuration identifier; weights not downloaded, revision not pinned |
| Pillow | 11.0.0, diagram renderer only | Four PNG/SVG pairs generated |

The native Windows machine has a GPU visible to Ray, but the selected PyTorch wheel is CPU-only. That is not a usable GPU-training environment. GPU scheduling declarations do not resolve driver or framework compatibility.

## API status

- Ray Serve LLM and Ray Data LLM: release 2.59.0 explicitly promotes them to GA/stable. LLMConfig and build_openai_app have stable public annotations in the installed source.
- TorchTrainer V2: stable public annotation in the 2.59.0 source. Examples enable RAY_TRAIN_V2_ENABLED=1 before importing Train.
- Old trainer restore/can_restore APIs: deprecated according to the tagged recovery guide; not used.
- Windows support: beta. Multi-node Windows support: experimental. Multi-node instructions target Linux.
- Ray Compiled Graph: documented as beta; not used.
- Ray Sandbox: experimental in release history; not used.
- FSDP2: PyTorch API, not an automatic Ray scheduling feature. Reviewed against PyTorch 2.9 documentation; not runtime-tested here.
- Runtime recovery is not established merely by stable API status. The failed Gloo setup prevented exercising the training loop and checkpoint recovery.

requirements.txt pins direct dependencies. requirements-authoring-windows.lock captures the successful CPU authoring environment. Neither file is a validated CUDA or Linux lockfile. requirements-llm.txt is a version-matched GPU configuration candidate. Run the resolver and pip check in that separate Linux environment before use.
