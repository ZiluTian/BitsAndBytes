# BitsAndBytes

Technical notes and traces.

## Docs

| Doc | What it covers |
|---|---|
| [How `conv2d` is lowered on Google TPU](torchtpu-conv2d-lowering.md) | File-level trace of one `F.conv2d` through TorchTPU: ATen dispatcher → deferred op → StableHLO emission (`stablehlo.convolution` + bias `add`) → XLA compile (open StableHLO→HLO / closed TPU backend) → PjRT runtime. Verified against `google-pytorch/torch_tpu` @ `main` and `openxla/xla` @ `main`, 2026-11-01. |
| [How `conv2d` is lowered on a GPU (cuDNN)](cudnn-conv2d-lowering.md) | File-level trace of one `F.conv2d` through PyTorch's native CUDA backend: ATen funnel → `aten::cudnn_convolution` (no bias; bias is a separate `add_`) → cuDNN v8 Frontend API (default) / v7 classic API (env toggle) → closed `libcudnn` kernel selection (implicit GEMM / Winograd / FFT / direct) → CUDA stream. Covers the recent `native/cuda/` → `native/cudnn/` refactor. Verified against `pytorch/pytorch` @ `cb9eb4b` (2026-11-01). |
| [`conv2d` on GPU vs TPU, side by side](cuda-vs-tpu-conv2d-comparison.md) | Same ATen funnel, two roads at the `convolution` dispatch: in-tree CUDA-key kernel calling closed cuDNN per op (no IR, per-shape algo cache) vs out-of-tree TorchTPU kernel deferring to StableHLO, compiled per graph by closed libtpu (192-bit key, 3-tier cache). 15 comparison axes (IR, compile unit, caching, layout, bias, backward, precision, determinism, closed stage, observability, …) + "same thing at different levels" observations. |
