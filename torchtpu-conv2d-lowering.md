# How `conv2d` is lowered on Google TPU (TorchTPU)

> A file-level trace of a single `F.conv2d` call — from Python down to the TPU
> silicon — through **TorchTPU** (Google's native PyTorch backend for TPU,
> open-sourced at the October 2026 PyTorch Conference).
>
> Verified against the actual source of
> [`google-pytorch/torch_tpu`](https://github.com/google-pytorch/torch_tpu)
> (branch `main`) and
> [`openxla/xla`](https://github.com/openxla/xla) (branch `main`)
> on **2026-11-01**. Line numbers refer to those revisions and will drift as
> the repos evolve — the symbols and file paths are the stable anchors.

---

## TL;DR — the pipeline

```
F.conv2d(x, w, b)            x, w, b on device="tpu" (NCHW, bf16)
   │
   ▼  PyTorch core (no TorchTPU code): autograd + native conv kernels
   │   all funnel into the generic op
aten::convolution            ATen dispatcher routes to the PrivateUse1 backend
   │
   ▼  STAGE 1 — TorchTPU ATen kernel
   │   records a "deferred op": an IR-builder lambda + cache keys.
   │   NOTHING is compiled yet.
   ▼  STAGE 2 — materialization (.cpu() / .item() / host sync)
   │   the deferred-op graph becomes one MLIR module (func.func @main)
   │   of stablehlo ops; fingerprinted into a 3-tier compilation cache
   ▼  STAGE 3 — xla::PjRtClient::Compile(module)
   │
   ├─ STAGE 4a — StableHLO → HLO        open source (openxla/xla), 1:1 mapping
   ├─ STAGE 4b — HLO → TPU executable   CLOSED SOURCE (inside libtpu)
   ▼  STAGE 5 — PjRtLoadedExecutable → PjRT ExecuteAsync
        HBM in, matmuls on the MXU, results back to host
```

The single most important architectural fact: **TorchTPU does not compile per
op.** Each op is captured as a *deferred* IR-builder; compilation happens once,
at a materialization point, for the whole captured graph. The TPU-specific
compilation itself is closed source — you can only observe it through HLO dumps.

---

## Stage 0 — PyTorch core (no TorchTPU code)

`torch.nn.functional.conv2d` goes through autograd. All of ATen's native
`conv1d/2d/3d` kernels funnel into the generic `aten::convolution` op (and
`aten::convolution_backward` in reverse). Because the tensors live on a
`PrivateUse1` device, the dispatcher selects TorchTPU's kernels instead of
cuDNN/MKL. Nothing below Stage 1 is TPU-specific.

---

## Stage 1 — Dispatcher registration + kernel → *deferred op*

### 1a. Registration

TorchTPU registers its conv kernels in the `PrivateUse1` dispatch key:

```cpp
// csrc/eager/tpu_aten_kernels.cc, inside TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) (L371)
ImplStable<OpName::kConvolution>(m, AtenConvolution);            // L473
ImplStable<OpName::kConvolutionBackward>(m, AtenConvolutionBackward); // L474
ImplStable<OpName::kConvolutionOut>(m, AtenConvolutionOut);      // L475
```

Source: [csrc/eager/tpu_aten_kernels.cc#L473](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc#L473)

### 1b. The kernel: parameter preparation

The `AtenConvolution` path lands in `ConvolutionOut`, which runs
`PrepareConvolutionDispatch` — a sequence of pure shape/param transforms that
re-do on the backend side everything PyTorch's CPU/CUDA funnels assume:

| Step | What it does | Source |
|---|---|---|
| `ExpandIfNecessary` (L76) | scalar `stride=2` → per-spatial-dim `[2, 2]` (PyTorch's broadcasting rule) | [convolution_aten_kernels.cc#L76](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L76) |
| shape/param validation | sanity checks on input/weight/stride/padding | [convolution_checks.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_checks.cc) |
| `GetOutputDimensions` (L130) | computes `(N, C_out, H_out, W_out)` with PyTorch's conv formula | [convolution_aten_kernels.cc#L130](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L130) |
| `GetPromotedType` (L183) | dtype promotion; rejects bool/long; input/weight/bias must match | [convolution_aten_kernels.cc#L183](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L183) |
| precision → cache key | reads the thread-local `torch.tpu.precision` context (`DEFAULT`/`HIGH`/`HIGHEST`) and folds it into the **parameter cache keys**, so changing precision invalidates the compilation cache | [convolution_aten_kernels.cc#L227](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L227) (`PrepareConvolutionDispatch`) |

Then it builds an **`op_builder` lambda** (L321–341) that captures the expanded
stride/padding/dilation, `output_dims`, the MLIR dtype, and the precision,
whose body is:

```cpp
return BuildConvolution(input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups, output_dims,
                        mlir_dtype, current_precision);
```

Source: [convolution_aten_kernels.cc#L352](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L352) (`ConvolutionOut`)

### 1c. Deferral

The lambda is handed to `DispatchOpOut<3>` → `DispatchOp`
([csrc/eager/op_dispatcher.h#L373](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/op_dispatcher.h#L373))
→ `CreateDeferredDeviceBufferList`
([csrc/eager/device_buffer_utils.cc#L231](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L231)):

```cpp
DeviceBufferList::CreateDeferred(op_name, std::move(op_builder), inputs,
                                 param_keys, output_shapes, ...)
```

**At this point nothing is compiled and no StableHLO exists.** You have a
`DeferredOp`: a record of (op name, builder lambda, input `DeviceBufferRef`s,
cache keys, output shape).

The *eager mode* decides what happens next — a switch at
[device_buffer_utils.cc#L240](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L240):

| Mode | How you get it | Behavior for your conv |
|---|---|---|
| `kDeferNever` (**default**) | — | materialize immediately: compile + launch a single-op program, asynchronously (the GPU-eager analogue) |
| `kDeferNeverAndLaunchBlocking` | `TORCH_TPU_LAUNCH_BLOCKING=1` | per-op `MaterializeCurrentStream(...).Await()` + `Synchronize()` + bounds checks (debug mode) |
| `kDeferAndFuse` | `TORCH_TPU_DEFER_AND_FUSE=1` | just `RecordDeferredOpCreated(...)`; stays deferred until a materialization point; a repeated-ops heuristic can trigger early background materialization when it detects repeated op patterns (e.g. a stack of transformer blocks) |

Mode resolution: [csrc/eager/eager_mode.cc#L34](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/eager_mode.cc#L34)

---

## Stage 2 — Materialization: assembling the StableHLO module

At a **materialization point** (`.cpu()`, `.item()`, any host sync), the
materialization pass (in [csrc/eager/materialize.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/materialize.cc)):

1. **Liveness-analyzes** the deferred-op DAG (side-effecting ops such as
   collectives are always kept; order preserved).
2. **Computes the 192-bit compilation cache key** — shapeless graph key +
   dimensions key + compile-options key, built from the `param_keys` collected
   in Stage 1 (which is why `torch.tpu.precision` changes the key) — and checks
   the **3-tier compilation cache**:
   - tier 1: in-process
   - tier 2: `/dev/shm/torch_tpu_cache/` (`TORCH_TPU_TIER2_COMPILATION_CACHE`)
   - tier 3: remote/GCS (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)
3. **On a miss**, `Traversal::BuildMlirModule`
   ([csrc/eager/traversal.cc#L597](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L597))
   builds the module — this is where your conv's lambda finally *runs*:

```cpp
mlir::ModuleBuilder mb(mlir_context, module_name_str);
mlir::func::FunctionBuilder fb(mb, "main");          // func.func @main
// one argument op per input DeviceBufferRef
for (const DeviceBufferRef& argument : arguments)
  ref_to_op_map[argument] = CreateArgumentOp(fb, argument, ...);
// then, for each deferred op in execution order:
//   call deferred_op.builder() with its inputs' MlirOps
//   → BuildConvolution emits the stablehlo.convolution etc.
```

(the `func.func @main` construction is at [traversal.cc#L616](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L616))

4. The finished MLIR module goes to `CompileAndLoadMlirModule` →
   `xla::PjRtClient::Compile`
   ([csrc/common/compilation.cc#L200](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L200),
   [compilation.cc#L224](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L224)).
   One TPU-specific quirk right there (comment at L229): XLA:TPU caps the
   number of program parameters (~16K on v7), so large training steps get
   their arguments **tupled** the way JAX does.

---

## Stage 3 — What `conv2d` actually emits: `stablehlo.convolution` + `stablehlo.add`

`BuildConvolution` in [csrc/ops/convolution/convolution.cc#L238](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L238)
(for `x:(N,Cin,H,W)`, `w:(Cout,Cin/G,kH,kW)`, `b:(Cout,)`):

```cpp
// NCHW dimension numbers — note: kernel output-feature is dim 0,
// i.e. PyTorch's Cout-first weight layout is used directly, no transpose
stablehlo::ConvDimensionNumbersAttr::get(&ctx,
    /*inputBatch=*/0,  /*inputFeature=*/1, {2, 3},
    /*kernelInputFeature=*/1, /*kernelOutputFeature=*/0, {2, 3},
    /*outputBatch=*/0,  /*outputFeature=*/1, {2, 3});

const int64_t feature_group_count = groups;          // L295
const int64_t batch_group_count   = 1;               // L297 — Torch groups over features only
const auto window_strides = DenseI64ArrayAttr::get(&ctx, stride);
// PyTorch symmetric padding → StableHLO asymmetric 2D [[p,p],[p,p]]  (L362-374)
const auto lhs_dilation  = {};                       // L306 — Torch only dilates the filter
const auto rhs_dilation  = DenseI64ArrayAttr::get(&ctx, dilation);
const auto window_reversal = {};                     // L309
stablehlo::Precision precisions[2] = {precision, precision};  // from torch.tpu.precision
const auto precision_config = PrecisionConfigAttr::get(&ctx, precisions);

input  = stablehlo::ConvertElementType(input, output_dtype);   // L318
weight = stablehlo::ConvertElementType(weight, output_dtype);  // L319

conv_op = ConvolutionLike(result_type, input, weight, dimension_numbers,
                          groups, 1, window_strides, dims_padding,
                          lhs_dilation, rhs_dilation, window_reversal,
                          precision_config);
// L339-344: bias is a SEPARATE op, not folded into the conv:
bias        = stablehlo::ConvertElementType(*bias, output_dtype);
shaped_bias = BroadcastIfNeeded(*bias, conv_op, /*broadcast_dimensions=*/{1});
return stablehlo::Add(conv_op, shaped_bias);
```

The actual emitter is `ConvolutionLike`
([csrc/ops/op_builder_utils.cc#L814](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc#L814)),
which calls `stablehlo::Convolution(...)`:

```cpp
return stablehlo::Convolution(static_result_type, lhs, rhs, dimension_numbers,
    feature_group_count, batch_group_count, window_strides, padding,
    lhs_dilation, rhs_dilation, window_reversal, precision_config);
```

So a forward conv2d becomes exactly:

```mlir
%0 = stablehlo.convert %input   : tensor<...> to tensor<...>
%1 = stablehlo.convert %weight
%2 = stablehlo.convolution %0, %1
     dim_numbers = [0, 1, 2, 3] x [1, 0, 2, 3] x [0, 1, 2, 3],
     feature_group_count = 1, batch_group_count = 1,
     window = { strides = [1, 1],
                padding = [[1, 1], [1, 1]],
                rhs_dilation = [1, 1] },
     precision_config = <DEFAULT>
%3 = stablehlo.broadcast_in_dim %bias, dims = [1]
%4 = stablehlo.add %2, %3
```

Notably there is **no im2col / matmul expansion here** — that happens later,
inside the (closed) TPU passes.

### 3a. Backward: all three gradients are convolutions

`AtenConvolutionBackward` (dispatch at
[convolution_aten_kernels.cc#L460](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L460))
emits three builders in [convolution.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc):

| Gradient | How it's built | Source |
|---|---|---|
| **grad_input** | an *adjoint* conv: `window_reversal = [true, ...]`, `lhs_dilation = stride` (the forward stride becomes the input dilation), asymmetric padding from `CalculateAdjointPadding` (L64: `pad_total = out + k_eff − 1 − ((in−1)·stride + 1)`) | [convolution.cc#L380](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L380) (`BuildConvolutionBackwardInput`) |
| **grad_weight** | a conv contracting `input × grad_output` over the batch dim (`batch_group_count = groups`), with a deliberate dimension-numbers trick (`outputBatchDimension=1, outputFeatureDimension=0`) so the result comes out **already in PyTorch's `(Cout, Cin/G, kH, kW)` weight layout** — the comment at L608–615 says this eliminates a post-hoc transpose copy for memory-bandwidth reasons | [convolution.cc#L572](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L572) (`BuildConvolutionBackwardWeight`) |
| **grad_bias** | a `stablehlo.reduce` (sum) over batch + spatial, promoted to a higher *computation dtype* (`InferComputationDtype`) then converted back | [convolution.cc#L684](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L684) (`BuildConvolutionBackwardBias`) |

The entire backward pass of conv is expressible with `stablehlo.convolution` +
one `stablehlo.reduce`.

### 3b. Two special paths

- **Bounded-dynamic (SymInt) shapes** — [op_builder_utils.cc#L837](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc#L837):
  XLA rejects bounded-dynamic spatial dims on conv, so `ConvolutionLike`
  zero-pads the operands at their upper bounds, runs the conv statically at the
  bounds, and `BoundConvolutionResult` crops to the runtime sizes.
- **Transposed conv** —
  [convolution.cc#L99](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L99)
  (`BuildTransposedConvolution`): a `window_reversal = true` conv with the
  weight transposed on the fly (reshape/transpose/reshape for `groups > 1`).

---

## Stage 4 — XLA compile (where the "real" TPU lowering happens)

### 4a. StableHLO → HLO — *open source*

In `openxla/xla`:

```cpp
// xla/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h (L76)
MAP_STABLEHLO_TO_HLO(ConvolutionOp)
```

Source: [map_stablehlo_to_hlo_op.h#L76](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h#L76) —
a 1:1 mapping into the HLO `Convolution` instruction; the converter lives in
[stablehlo_legalize_to_hlo.cc](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo.cc).

### 4b. HLO → TPU executable — *closed source*

⚠️ **The TPU backend is no longer in the public XLA repo.** As of the
2026-11-01 `main` tree (verified against the full git tree — 10,693 paths,
untruncated), `xla/backends/` contains only
[cpu](https://github.com/openxla/xla/tree/main/xla/backends/cpu),
[gpu](https://github.com/openxla/xla/tree/main/xla/backends/gpu),
[interpreter](https://github.com/openxla/xla/tree/main/xla/backends/interpreter),
and [profiler](https://github.com/openxla/xla/tree/main/xla/backends/profiler).
(The Mosaic TPU *dialect* and the StableHLO/HLO machinery remain open; the
compiler backend itself does not.)

So layout assignment, TPU instruction fusion, the **conv → matmul expansion**
(the MXU is a matmul unit — convolutions end up as `dot`/implicit-GEMM with
reshape/transpose), tiling, and codegen to the proprietary TPU ISA all happen
inside `libtpu`. You can only observe its *output*. The executable is a
`PjRtLoadedExecutable`, serialized into tiers 2/3 of the compilation cache.

---

## Stage 5 — PjRT runtime

[csrc/pjrt/pjrt_execute.cc](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/pjrt/pjrt_execute.cc) —
`ExecuteAsync`: the PjRT client launches the executable on the TPU with the
argument HBM buffers; execution is asynchronous (the CPU keeps running), the
result lives in a `PjRtBuffer`; `.cpu()` waits on the buffer and DMAs
HBM → host through the premapped buffer (`TPU_PREMAPPED_BUFFER_SIZE`).

---

## Observing every stage yourself

No GCP account needed — Colab's free **v5e** TPU runtime is listed under
"Validated" in the [TorchTPU feature-set docs](https://torchtpu.readthedocs.io/):

```python
import os
dump = "/tmp/hlo_dump"
os.makedirs(dump, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump}"
# optional:
# os.environ["TORCH_TRACE"] = "/tmp/torch_trace"          # dispatcher-level structured trace
# os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"            # fused eager regions instead of per-op

import torch
x = torch.randn(4, 8, 32, 32, device="tpu", dtype=torch.bfloat16)
w = torch.randn(16, 8, 3, 3,  device="tpu", dtype=torch.bfloat16)
b = torch.randn(16,           device="tpu", dtype=torch.bfloat16)

y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
print(y.sum().cpu().item())   # materialization point → compile happens here
```

In `/tmp/hlo_dump` you then get:

- **`*.after_optimizations.txt`** — the *final* HLO after the (closed) TPU
  passes. For the conv above you see the `convolution` instruction rewritten
  into `dot` (matmul) + `reshape`/`transpose` with the bias `add` (often
  fused), plus concrete HBM buffer layouts and tiling. **This file *is* the
  observable end-state of Stage 4.**
- `*.flagfile` — active compiler flags (optimization level, etc.).
- `*tpu_comp_env.txt` — TPU compile-environment settings.

The dump flags live in
[xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc#L1889)
(`--xla_dump_hlo_as_text`, `--xla_dump_to`); the dump machinery is in
[xla/service/dump.cc](https://github.com/openxla/xla/blob/main/xla/service/dump.cc).
Note the current public XLA debug flags dump *HLO*, not StableHLO — but the
StableHLO→HLO map for convolution is 1:1, so the pre-optimization HLO dump
effectively shows you the emitted `stablehlo.convolution`.

---

## Quirks worth knowing (conv-specific)

1. **Bias is a separate `stablehlo.add`** — not absorbed into the conv op;
   whether it survives as a fused kernel is the (closed) XLA TPU passes'
   decision.
2. **Precision is a compile-cache dimension** — `torch.tpu.precision(HIGH)`
   changes `precision_config` *and* the cache key, so the first call under a
   new precision recompiles.
3. **NCHW is native** — TorchTPU emits NCHW dim numbers directly (kernel
   `output_feature` = dim 0); on CUDA you'd typically see cuDNN working in
   channels-last.
4. **The weight-gradient trick** — `batch_group_count` + flipped dim numbers
   produce `grad_weight` already in PyTorch layout (no transpose buffer).
5. **Transposed conv is a different animal** — `window_reversal = true` with
   an on-the-fly weight transpose, not a separate op.

---

## Source index

### `google-pytorch/torch_tpu` (branch `main`)

| Concern | File | Key lines |
|---|---|---|
| `PrivateUse1` kernel registration | [`csrc/eager/tpu_aten_kernels.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc) | [L371](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc#L371) (block), [L473–475](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/tpu_aten_kernels.cc#L473) (conv) |
| ATen conv kernel (forward) | [`csrc/ops/convolution/convolution_aten_kernels.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc) | [L76](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L76), [L130](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L130), [L183](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L183), [L227](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L227), [L352](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L352) |
| ATen conv kernel (backward dispatch) | same file | [L460–510](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_aten_kernels.cc#L460) |
| StableHLO conv builders | [`csrc/ops/convolution/convolution.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc) | [L64](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L64), [L99](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L99), [L238](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L238), [L380](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L380), [L572](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L572), [L684](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution.cc#L684) |
| conv param validation | [`csrc/ops/convolution/convolution_checks.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/convolution/convolution_checks.cc) | — |
| `stablehlo.convolution` emitter + bounded-dynamic path | [`csrc/ops/op_builder_utils.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc) | [L814](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc#L814), [L837](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/ops/op_builder_utils.cc#L837) |
| Op dispatch / deferred-op creation | [`csrc/eager/op_dispatcher.h`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/op_dispatcher.h) | [L373](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/op_dispatcher.h#L373) |
| Deferred buffers + eager-mode switch | [`csrc/eager/device_buffer_utils.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc) | [L231](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L231), [L240](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/device_buffer_utils.cc#L240) |
| Eager-mode resolution | [`csrc/eager/eager_mode.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/eager_mode.cc) | [L34–40](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/eager_mode.cc#L34) |
| Materialization | [`csrc/eager/materialize.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/materialize.cc) | — |
| `func.func @main` module build | [`csrc/eager/traversal.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc) | [L597](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L597), [L616](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/eager/traversal.cc#L616) |
| XLA compile handoff | [`csrc/common/compilation.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc) | [L200](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L200), [L224](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L224), [L229](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/common/compilation.cc#L229) (param-cap comment) |
| PjRT execution | [`csrc/pjrt/pjrt_execute.cc`](https://github.com/google-pytorch/torch_tpu/blob/main/csrc/pjrt/pjrt_execute.cc) | — |

### `openxla/xla` (branch `main`)

| Concern | File |
|---|---|
| StableHLO→HLO op map (`ConvolutionOp` at L76) | [xla/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h#L76) |
| StableHLO→HLO legalization pass | [xla/mlir_hlo/mhlo/transforms/stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo.cc](https://github.com/openxla/xla/blob/main/xla/mlir_hlo/mhlo/transforms/stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo.cc) |
| Dump flags (`--xla_dump_hlo_as_text`, `--xla_dump_to`) | [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc#L1889) |
| Dump machinery | [xla/service/dump.cc](https://github.com/openxla/xla/blob/main/xla/service/dump.cc) |
| Backends present in the public repo (cpu, gpu, interpreter, profiler — **no tpu**) | [xla/backends/](https://github.com/openxla/xla/tree/main/xla/backends) |

---

*Caveats: (1) line numbers were correct as of 2026-11-01 and will drift;
symbol names and file paths are the durable anchors. (2) Stage 4b is closed
source by design — every claim in that section is about what is (or is not)
publicly observable, not about the closed code itself.*
