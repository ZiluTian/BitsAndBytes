# How `conv2d` is lowered: GPU (cuDNN) vs TPU (TorchTPU/XLA), side by side

> A comparison of the two companion traces:
> [cudnn-conv2d-lowering.md](cudnn-conv2d-lowering.md) (PyTorch → cuDNN, verified
> against `pytorch/pytorch` @ `cb9eb4b`, 2026-11-01) and
> [torchtpu-conv2d-lowering.md](torchtpu-conv2d-lowering.md) (TorchTPU → XLA,
> verified against `google-pytorch/torch_tpu` @ `main` and `openxla/xla` @ `main`,
> 2026-11-01). Line numbers refer to those revisions; symbol names and file paths
> are the stable anchors.

---

## TL;DR — one funnel, two roads

```
F.conv2d(x, w, b)
   │
   ▼  SHARED: PyTorch core (no backend code)
aten::conv2d → conv2d_symint (Convolution.cpp:1038) → generic aten::convolution (yaml L1680)
   │
   ╔══════════════════════════════════╗  ║  ╔══════════════════════════════════════╗
   ║ device = "cuda"                  ║  ║  ║ device = "tpu" (PrivateUse1)         ║
   ║ in-tree CUDA-key kernel           ║  ║  ║ out-of-tree TorchTPU kernel          ║
   ║  _convolution (Convolution.cpp    ║  ║  ║  AtenConvolution                    ║
   ║   :1576)                          ║  ║  ║  (tpu_aten_kernels.cc:473)          ║
   ║   → _select_conv_backend (L1297)  ║  ║  ║   → ConvolutionOut (L352)           ║
   ║   → ConvBackend::Cudnn (L1638)    ║  ║  ║   → DispatchOpOut                   ║
   ║ aten::cudnn_convolution           ║  ║  ║   → CreateDeferred                  ║
   ║ (yaml L1885 — NO bias arg)        ║  ║  ║  = DEFERRED OP (nothing compiled)   ║
   ║ + output.add_(bias) (L1643)       ║  ║  ║                                      ║
   ║   │                               ║  ║  ║   │                                  ║
   ║   ▼ PER-OP:                       ║  ║  ║   ▼ AT A MATERIALIZATION POINT      ║
   ║   closed libcuDNN picks a         ║  ║  ║   (.cpu() / .item() / host sync):    ║
   ║   prebuilt kernel (implicit GEMM /║  ║  ║   deferred graph → MLIR module      ║
   ║   Winograd / FFT / direct),       ║  ║  ║   (func.func @main of stablehlo ops)║
   ║   launches it on the current CUDA ║  ║  ║   → 192-bit cache key → 3-tier      ║
   ║   stream; algo cached in-process  ║  ║  ║   cache (proc → /dev/shm → GCS)     ║
   ║   → NO intermediate IR            ║  ║  ║   miss → xla::PjRtClient::Compile   ║
   ║                                   ║  ║  ║   (CLOSED libtpu) → executable →    ║
   ║                                   ║  ║  ║   PjRT ExecuteAsync (HBM, MXU)      ║
   ╚══════════════════════════════════╝  ║  ╚══════════════════════════════════════╝
```

The two roads diverge at exactly one point: **the implementation of the generic
`aten::convolution` op**. For a CUDA tensor the in-tree composite kernel runs
(`_convolution`, which contains a backend switch and calls the closed cuDNN
library *per op* — there is no IR, no graph, only "which prebuilt kernel").
For a TPU tensor, TorchTPU's out-of-tree `PrivateUse1` kernel takes over and,
instead of computing, *records IR* — the graph is then compiled once, at a
materialization point, by the closed XLA TPU compiler into a machine-code
executable.

In one sentence each:

- **GPU**: per-op dispatch into a closed runtime library that *selects among
  prebuilt kernels*.
- **TPU**: deferred IR capture, compiled per *graph* by a closed graph
  compiler that *writes the code*.

---

## The shared funnel (where they're identical)

Both paths start in PyTorch core, with zero backend-specific code:

1. `torch.nn.functional.conv2d` → autograd wraps `aten::conv2d`
   ([native_functions.yaml#L1719](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1719),
   `CompositeImplicitAutograd: conv2d_symint`).
2. `conv2d_symint`
   ([Convolution.cpp#L1038](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1038))
   — dtype checks, `batchify`, complex path — calls the generic
   `aten::convolution`
   ([native_functions.yaml#L1680](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1680)).

That generic op is the fork. Two dispatch keys, two worlds:

| | CUDA road | TPU road |
|---|---|---|
| Who implements `aten::convolution` | the in-tree composite kernel | TorchTPU's `PrivateUse1` kernel, [tpu_aten_kernels.cc#L473](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc#L473) |
| What it does | computes: `_convolution` ([Convolution.cpp#L1576](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1576)) → `_select_conv_backend` (L1297) → `ConvBackend::Cudnn` (L1638) → `at::cudnn_convolution` (yaml [L1885](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1885)) + `output.add_(bias)` (L1643) | records: `ConvolutionOut` (convolution_aten_kernels.cc:352) → `DispatchOpOut` → `DeviceBufferList::CreateDeferred` ([device_buffer_utils.cc#L231](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L231)) — a deferred op (IR-builder lambda + cache keys), nothing compiled |
| What the op signature looks like | `cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32)` — **no bias** | `BuildConvolution` emits `stablehlo.convolution` + `stablehlo.add` (bias is its own IR op) |

Everything below is the comparison axis by axis.

---

## Axis 1 — Where the code lives

| | GPU | TPU |
|---|---|---|
| Backend implementation | **In-tree**: `aten/src/ATen/native/Convolution.cpp`, `aten/src/ATen/native/cudnn/{ConvShared.cpp, Conv_v7.cpp, Conv_v8.cpp}`, `aten/src/ATen/cudnn/Descriptors.h` | **Out-of-tree**: `google-pytorch/torch_tpu` (`csrc/eager/`, `csrc/ops/convolution/`, `csrc/pjrt/`) |
| Open IR layer | none — cuDNN descriptors / v8 op graphs are API objects, not IR | StableHLO → HLO, open in `openxla/xla` (1:1 conv map at [map_stablehlo_to_hlo_op.h#L76](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h#L76)) |
| Closed component | `libcudnn` (+ `libcudnn_cnn_infer`) — a **runtime kernel library** | `libtpu` — a **graph compiler**; the TPU backend is no longer in the public `openxla/xla` repo (`xla/backends/` = cpu, gpu, interpreter, profiler only, verified 2026-11-01) |

On the GPU road the *entire open-source half* is the glue: ATen funnel,
backend switch, descriptor construction, algo-cache bookkeeping. On the TPU
road the open-source half is a *backend that speaks IR*: parameter
preparation, StableHLO emission, cache keying, PjRT plumbing.

---

## Axis 2 — Intermediate representation

- **GPU: none.** The closest artifact is the cuDNN *op graph* in the v8 path —
  a single node, `CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`
  ([Conv_v8.cpp#L1483](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1483)) — or, in the v7 path, the
  descriptor triple (tensor/filter/conv). These are *API handles*, not
  something you can dump, hash, or diff.
- **TPU: real IR.** Materialization builds an MLIR module —
  `func.func @main` of `stablehlo` ops
  ([traversal.cc#L597](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L597),
  [L616](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L616)) —
  which you can print (`--xla_dump_hlo_as_text`), fingerprint (the 192-bit
  cache key *is* a fingerprint of it), and diff. Your forward conv is literally
  the five-line `stablehlo.convert ×2 / convolution / broadcast_in_dim / add`
  module shown in the [TPU doc](torchtpu-conv2d-lowering.md).

Consequence: on the TPU road, *the program is the artifact*; on the GPU road,
*the data layout + algo number* are the artifacts.

---

## Axis 3 — Unit of compilation / work

| | GPU | TPU |
|---|---|---|
| Unit | **one op, per call** | **one graph, per materialization** |
| What "running" means | pick an algorithm (or replay a cached one) and launch a prebuilt kernel on the stream | `xla::PjRtClient::Compile(module)` → `PjRtLoadedExecutable` → `PjRT ExecuteAsync` |
| Compile cost | heuristic query (fast) or exhaustive benchmark (`benchmark=True`, measured timings) | a full XLA TPU compilation — the expensive step; the whole reason the 3-tier cache exists |

A nuance that matters: TorchTPU's *default* eager mode (`kDeferNever`,
[device_buffer_utils.cc#L240](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L240))
still compiles *per op* — it materializes each conv immediately and compiles a
single-op program, asynchronously. So "eager on TPU" looks like eager on CUDA
from Python's perspective, but each eager op is a *compiled program*, not a
kernel launch. `TORCH_TPU_DEFER_AND_FUSE=1` is what turns a repeated region
into one multi-op executable (the true "graph" mode).

---

## Axis 4 — Caching

| | GPU | TPU |
|---|---|---|
| Key | v7: `ConvolutionParams` (shapes + params + benchmark/deterministic/TF32) in an in-process `BenchmarkCache` ([Conv_v7.cpp#L105](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L105)); v8: a `CacheKeyWrapper` over (operation, output, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32) ([Conv_v8.cpp#L1165](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1165)) | 192-bit key = shapeless graph key + dimensions key + compile-options key, built from each op's `param_keys` |
| What invalidates it | new shape, new flags (`benchmark`/`deterministic`/`allow_tf32`) | new graph structure, new dims, *new precision* (`torch.tpu.precision` is folded into `param_keys` — [convolution_aten_kernels.cc#L227](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L227)) |
| Lifetime | dies with the process | **3 tiers**: in-process → `/dev/shm/torch_tpu_cache/` → remote/GCS — survives restarts, shared across processes and machines |
| First-call cost | heuristic query, or a measured benchmark sweep if `benchmark=True` | a full XLA compilation — amortized across every later process on that machine (and every machine that shares the tier-3 root) |

This is the most practically different axis: cuDNN's cache answers
"which kernel"; TorchTPU's cache answers "which *executable*", and it's the
difference between a ~ms warmup and a minutes-scale warmup on a cache miss.

---

## Axis 5 — How the convolution is represented

**GPU** — cuDNN descriptors (v7) / backend op graph (v8):

- tensor descriptors carry the **actual memory strides**, so layout is a
  *data property*: NCHW strides or NHWC strides
  ([Descriptors.h#L169](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L169))
- the conv descriptor is
  `cudnnSetConvolutionNdDescriptor(dim, pad, stride, dilation, CUDNN_CROSS_CORRELATION, mathType)`
  + `cudnnSetConvolutionGroupCount(groups)` + initial math mode
  ([Descriptors.h#L203](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L203))

**TPU** — one `stablehlo.convolution` op with explicit semantics:

```mlir
%2 = stablehlo.convolution %0, %1
     dim_numbers = [0, 1, 2, 3] x [1, 0, 2, 3] x [0, 1, 2, 3],
     feature_group_count = 1, batch_group_count = 1,
     window = { strides = [1, 1], padding = [[1, 1], [1, 1]], rhs_dilation = [1, 1] },
     precision_config = <DEFAULT>
```

- **dim_numbers are explicit IR** (NCHW; kernel output-feature = dim 0, so
  PyTorch's Cout-first weight is used *as-is* — no transpose)
  ([convolution.cc#L238](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L238))
- groups are `feature_group_count = groups`, `batch_group_count = 1`
  (convolution.cc:295/297) — Torch groups over features only
- dilation is `rhs_dilation` only; `lhs_dilation` is empty (Torch dilates the
  filter, not the input)

**Cross-correlation, spelled two different ways.** cuDNN's default is a true
(flipped-kernel) convolution, so PyTorch must *opt out* with
`CUDNN_CROSS_CORRELATION` (Descriptors.h:207). StableHLO's default — empty
`window_reversal` — *is* cross-correlation, so the reversal appears only where
the math needs it: the backward adjoint
([convolution.cc#L380](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L380)).
Same intent, opposite directions.

---

## Axis 6 — Layout

- **GPU**: a *heuristic* you can influence. If the input **or** weight is
  channels-last and dtype ≠ float64, the conv runs in NHWC
  ([ConvUtils.h#L348](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L348),
  `cudnn_conv_suggest_memory_format` at L370) — and the *output* is allocated
  in that format, so it may be physically NHWC even though your code says
  NCHW. `model.to(memory_format=torch.channels_last)` changes kernel selection
  without touching the model.
- **TPU**: the IR says NCHW (dim numbers); the *physical* HBM layout is chosen
  by the closed TPU layout-assignment pass — visible only in the
  `after_optimizations` HLO dump.

So on the GPU you (or a heuristic) pick the layout *before* the closed stage;
on the TPU the closed stage picks it *for* you, inside the compile.

---

## Axis 7 — Where the bias goes

| | GPU (eager) | TPU |
|---|---|---|
| In the conv op? | **No** — `aten::cudnn_convolution` has no bias argument (yaml L1885) | **No** — `stablehlo.convolution` has no bias attribute |
| How it's added | a separate in-place `output.add_(reshape_bias(...))` — a distinct elementwise kernel (Convolution.cpp:1643) | a separate `stablehlo.broadcast_in_dim` + `stablehlo.add` in the *same IR module* (convolution.cc:339–344) |
| When it fuses | only in the TorchScript frozen-model pass (`frozen_conv_add_relu_fusion_cuda.cpp:39/58` → fused `cudnn_convolution_add_relu` graph op) | whenever the closed XLA passes choose to fuse `add` into the conv kernel |

The same shape at both levels: *at the level you can see, the bias is a
separate add*. The difference is what "separate" means — on the GPU it's a
runtime fact (two kernel launches in eager mode, confirmed by the torch
profiler showing `aten::cudnn_convolution` **and** `aten::add_`); on the TPU
it's an IR statement that the compiler may optimize away before codegen.

---

## Axis 8 — Backward

| | GPU | TPU |
|---|---|---|
| **grad_input** | dedicated dgrad call: `cudnnConvolutionBackwardData` ([Conv_v7.cpp#L930](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L930)) | an *adjoint conv* in IR: `window_reversal = [true, …]`, `lhs_dilation = stride`, asymmetric padding (`CalculateAdjointPadding`, convolution.cc:64) — [convolution.cc#L380](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L380) |
| **grad_weight** | dedicated wgrad call: `cudnnConvolutionBackwardFilter` ([Conv_v7.cpp#L1049](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L1049)) | a conv contracting `input × grad_output` over the batch dim, with a dim-numbers trick (`batch_group_count = groups`, flipped output dims) so the result lands **already in PyTorch's `(Cout, Cin/G, kH, kW)` layout** (convolution.cc:608–615) — [convolution.cc#L572](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L572) |
| **grad_bias** | *not from cuDNN* — the source comment says it plainly: "This function only exists because cuDNN does not support bias gradient computation" (FunctionsManual.cpp:7585–7589); grad_bias is a generic sum over batch + spatial (fallback at Convolution.cpp:2380) | a `stablehlo.reduce` (sum) over batch + spatial, promoted to a higher computation dtype — [convolution.cc#L684](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L684) |

Two observations:

- **grad_weight is "a convolution" on both sides.** On the GPU it's a
  prebuilt kernel that NVIDIA shipped; on the TPU it's an IR transform TorchTPU
  emits. Same adjoint mathematics, different packaging.
- The old PyTorch "role-swap" trick (compute wgrad as a *forward* conv with
  input/weight exchanged) is **not** what the CUDA path does — cuDNN provides
  a dedicated backward-filter call, so PyTorch just calls it.

---

## Axis 9 — Algorithm choice vs codegen

- **GPU**: cuDNN *selects* among **prebuilt kernel families** — implicit GEMM
  (the workhorse; the conv is an im2col-style GEMM on tensor cores), Winograd,
  FFT, direct — each with FP16/BF16/tensor-core variants, per SM architecture.
  Default = heuristics; `torch.backends.cudnn.benchmark = True` = exhaustive
  measured search (`cudnnFindConvolutionForwardAlgorithmEx`, Conv_v7.cpp:319
  / `get_plans_from_find`, Conv_v8.cpp:1282). The choice is cached per shape.
- **TPU**: the closed XLA passes *rewrite* the conv into
  `dot` + `reshape`/`transpose` — the MXU is a matmul unit, so convolutions
  end up as matmuls on the silicon — plus layout assignment and tiling, all
  per graph, all inside `libtpu`. There is no "algo number" to pick; the
  compiler writes the code.

So: on the GPU the interesting question is *which prebuilt kernel*; on the TPU
it's *what the compiler generated* (visible only in the post-optimization HLO
dump).

---

## Axis 10 — Precision

| | GPU | TPU |
|---|---|---|
| Knob | `torch.backends.cudnn.allow_tf32` (+ dtype) | `torch.tpu.precision` thread-local context (`DEFAULT` / `HIGH` / `HIGHEST`) |
| How it's expressed | math mode in the conv descriptor: fp16/bf16 → `CUDNN_TENSOR_OP_MATH` (tensor cores); fp32 → `CUDNN_FMA_MATH` when TF32 is disallowed, `CUDNN_DEFAULT_MATH` otherwise (Descriptors.h:210–215; the deterministic path at Conv_v7.cpp:566–576) | `precision_config` attribute on the `stablehlo.convolution` op (convolution.cc:238) |
| Cache effect | part of the per-shape algo key (`allow_tf32`) | part of the **192-bit graph key** (convolution_aten_kernels.cc:227) — changing precision under a running model *recompiles* |

---

## Axis 11 — Determinism

- **GPU**: `torch.backends.cudnn.deterministic` (env `CUDNN_DETERMINISTIC`)
  with `benchmark=False` forces `DEFAULT_ALGO` only (Conv_v7.cpp:566–576) —
  determinism costs performance, and the *algo search itself* is the
  non-determinism source you're pinning down.
- **TPU**: there is no separate knob. Compilation is a pure function of the
  cache key: same graph + same dims + same options → same cached executable.
  The question "did a different algorithm get picked?" doesn't arise.

---

## Axis 12 — The closed stage, and what kind of "closed" it is

This is the deepest difference:

| | GPU | TPU |
|---|---|---|
| Closed component | `libcudnn` — a **runtime library** | `libtpu` — a **compiler** |
| What it decides | *which* prebuilt kernel + its tiling/workspace, per call | the *entire machine-code program*, per graph: layout, fusion, conv→dot expansion, tiling, ISA codegen |
| State it holds | algo/plan caches (in-process on the PyTorch side; cuDNN has its own internal caches) | `PjRtLoadedExecutable` objects, serialized into the tier-2/3 compilation cache |
| What you can observe | kernel names (Nsight), candidate algos + timings (`CUDNN_LOGLEVEL_DEBUG`) | HLO dumps (pre- and post- the closed passes); the executable itself is opaque |

Both hide the last mile from you, but in different senses: the GPU's last mile
is *"pick a kernel"* (a data-driven runtime decision); the TPU's last mile is
*"compile a program"* (a code-generation decision). That single difference
drives the cache design (Axis 4), the observability tools (Axis 13), and the
first-call cost profile.

---

## Axis 13 — Observability, side by side

| Want to see | GPU | TPU |
|---|---|---|
| The actual "program" that ran | Nsight Systems kernel names: `smXX_xmma_tensorop_…_convndf_…` (implicit-GEMM fwd), `…wgrad…`/`…dgrad…` variants | the `*.after_optimizations.txt` HLO dump: `convolution` rewritten to `dot` + `reshape`/`transpose`, bias `add` (fused or not), HBM layouts, tiling |
| What the backend itself saw | `CUDNN_LOGLEVEL_DEBUG=4 CUDNN_LOGDEST_DBG=/tmp/cudnn.log` — the cuDNN graph, candidate algos, measured timings | `XLA_FLAGS=--xla_dump_hlo_as_text --xla_dump_to=/tmp/hlo_dump` (flags in [debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc)) |
| Python-level view | torch profiler: `aten::cudnn_convolution` **and** `aten::add_` nodes; exactly two device kernels for a biased conv | torch profiler: op-level nodes; the compile boundary is visible at materialization |
| Which algo/kernel was picked | v7 benchmark logs; `TORCH_CUDNN_V8_API_DEBUG=1` for the v7/v8 gate | not exposed as a number — infer from HLO structure |
| Force the "other" path | `TORCH_CUDNN_V8_API_DISABLED=1` (classic v7 API instead of Frontend API) | `TORCH_TPU_DEFER_AND_FUSE=1` (graph mode instead of per-op) |

The GPU's analogue of an "HLO dump" is the cuDNN debug log + Nsight; the TPU's
analogue of Nsight kernel names is the post-optimization HLO structure.

---

## Axis 14 — Large tensors and dynamic shapes

- **GPU**: the v7 path has an explicit workaround — if input/output numel
  exceeds int32, the batch dimension is split and the conv runs in slices
  ([Conv_v7.cpp#L673](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L673)); the v8 path
  instead expects cuDNN to handle large tensors (ConvShared.h:139–146).
- **TPU**: XLA rejects bounded-dynamic spatial dims on conv, so TorchTPU's
  `ConvolutionLike` zero-pads the operands to their upper bounds, runs the conv
  *statically* at the bounds, then crops to runtime sizes
  ([op_builder_utils.cc#L837](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc#L837)).

Both are "make the hard case fit the closed stage" workarounds — one splits
data at the ATen level, the other pads IR operands.

---

## Axis 15 — Runtime & asynchrony

- **GPU**: the cuDNN handle is bound to the *current* CUDA stream on every use
  (`cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream())`,
  [Handle.cpp#L53](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L53)).
  Kernels launch asynchronously; the CPU keeps running; synchronization is the
  CUDA runtime's business.
- **TPU**: `PjRT ExecuteAsync` launches the compiled executable with the
  argument HBM buffers ([pjrt_execute.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/pjrt/pjrt_execute.cc));
  execution is asynchronous; `.cpu()` waits on the resulting `PjRtBuffer` and
  DMAs HBM → host through a premapped buffer (`TPU_PREMAPPED_BUFFER_SIZE`).

Both are "fire and the CPU continues"; the difference is that on the TPU the
unit fired is a whole (possibly multi-op) executable, and the host-sync
*trigger* is also the compile trigger — materialization and compilation are
the same event.

---

## The same thing at different levels

A few points where the two roads are structurally the same:

1. **Bias lives outside the conv op on both roads** — a separate `add_` on
   CUDA, a separate `stablehlo.add` on TPU — and on *both* roads whether it
   fuses into the conv kernel is decided by the closed stage. The difference
   is only at what level "separate" is guaranteed (runtime, vs IR-maybe).
2. **`grad_weight` is a convolution on both roads** — a prebuilt wgrad kernel
   on CUDA, an IR adjoint on TPU. Same math, different packaging.
3. **Cross-correlation is explicit on both roads**, spelled in opposite
   directions: cuDNN defaults to a flipped kernel (opt out with
   `CUDNN_CROSS_CORRELATION`); StableHLO defaults to cross-correlation (opt in
   to reversal via `window_reversal`, used by the backward).
4. **Both roads end in a closed stage of a different kind** — a runtime
   library that *selects kernels* (cuDNN) vs a compiler that *writes code*
   (libtpu). Every other difference in this doc (caching, first-call cost,
   observability, determinism semantics) follows from that.
5. **Cache granularity and lifetime differ by orders of magnitude**: a
   per-shape algo cached in-process (GPU) vs a per-graph executable cached
   across processes and machines in 3 tiers (TPU).
6. **Layout**: strides-as-*data* (cuDNN tensor descriptors, chosen by a
   heuristic you can influence) vs dim-numbers-as-*IR* (StableHLO, with the
   physical HBM layout decided by the closed compiler).

---

## Caveats

1. Line numbers were correct at the pinned revisions (2026-11-01:
   `pytorch/pytorch` @ `cb9eb4b`, `google-pytorch/torch_tpu` @ `main`,
   `openxla/xla` @ `main`) and will drift; symbol names and file paths are the
   durable anchors.
2. Both traces cover the **native eager** path. `torch.compile` on CUDA
   (Triton / cuDNN graphs) and AOT-compiled XLA are different pipelines on
   both sides — this comparison does not apply to them.
3. On the GPU side, the **cuDNN v8 Frontend/Backend API is the default**; the
   classic v7 API is retained behind `TORCH_CUDNN_V8_API_DISABLED=1`. Both are
   traced in the [cuDNN doc](cudnn-conv2d-lowering.md).
4. Both closed stages (libcudnn, libtpu): every claim about them is about what
   is publicly *observable* (kernel names, debug logs, HLO dumps), not about
   their internals.

---

## Source anchors for this comparison

The full source indices are in the two companion docs
([cuDNN](cudnn-conv2d-lowering.md#source-index),
[TPU](torchtpu-conv2d-lowering.md#source-index)); the fork and the
comparison-critical lines:

| Concern | GPU (`pytorch/pytorch` @ `cb9eb4b`) | TPU (`google-pytorch/torch_tpu` @ `main`) |
|---|---|---|
| The fork: `aten::convolution` implementation | in-tree composite → `_convolution` ([Convolution.cpp#L1576](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1576)); backend switch (L1297); `Cudnn` case (L1638); bias `add_` (L1643) | out-of-tree kernel ([tpu_aten_kernels.cc#L473](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc#L473)); `ConvolutionOut` ([convolution_aten_kernels.cc#L352](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L352)) |
| The conv "op" that reaches the closed stage | `aten::cudnn_convolution` (yaml L1885) → [ConvShared.cpp#L243](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L243) → v8 [Conv_v8.cpp#L1464](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1464) / v7 [Conv_v7.cpp#L840](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L840) | deferred op ([device_buffer_utils.cc#L231](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L231)) → `stablehlo.convolution` + `stablehlo.add` ([convolution.cc#L238](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L238)) |
| Per-op vs per-graph unit | algo search + launch: v7 [Conv_v7.cpp#L582](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L582), v8 [Conv_v8.cpp#L1152](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1152) | materialize → module → compile: [traversal.cc#L597](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L597), [compilation.cc#L200](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L200) |
| Cache | in-process `BenchmarkCache` (Conv_v7.cpp:105) / plan cache (Conv_v8.cpp:1177) | 192-bit key + 3 tiers (in-process → `/dev/shm` → GCS), [materialize.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/materialize.cc) |
| Backward | `cudnnConvolutionBackwardData` (Conv_v7.cpp:930) / `…BackwardFilter` (L1049); bias-grad comment (FunctionsManual.cpp:7585) | adjoint convs + reduce: [convolution.cc#L380](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L380), [L572](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L572), [L684](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L684) |
| The closed stage | `libcudnn` (runtime kernel selection) | `libtpu` (graph compiler; TPU backend absent from public `openxla/xla`) |
| Observation | `CUDNN_LOGLEVEL_DEBUG`/`CUDNN_LOGDEST_DBG`, Nsight kernel names | `--xla_dump_hlo_as_text --xla_dump_to`, [debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc) |
