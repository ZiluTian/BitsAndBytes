# BitsAndBytes

Technical notes and traces.

## Docs

| Doc | What it covers |
|---|---|
| [How `conv2d` is lowered on Google TPU](torchtpu-conv2d-lowering.md) | File-level trace of one `F.conv2d` through TorchTPU: ATen dispatcher → deferred op → StableHLO emission (`stablehlo.convolution` + bias `add`) → XLA compile (open StableHLO→HLO / closed TPU backend) → PjRT runtime. Verified against `google-pytorch/torch_tpu` @ `main` and `openxla/xla` @ `main`, 2026-11-01. |
