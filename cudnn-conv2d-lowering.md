# How `conv2d` is lowered on a GPU (PyTorch → cuDNN)

> A file-level trace of a single `F.conv2d` call — from Python down to the CUDA
> stream — through PyTorch's native CUDA backend and the NVIDIA **cuDNN**
> runtime. The counterpart to
> [torchtpu-conv2d-lowering.md](torchtpu-conv2d-lowering.md); the two
> documents share Stage 0 and diverge at the `convolution` dispatch.
>
> Verified against the actual source of
> [`pytorch/pytorch`](https://github.com/pytorch/pytorch)
> (branch `main`, commit `cb9eb4b`, 2026-11-01). Line numbers refer to that
> revision and will drift — the symbols and file paths are the stable anchors.
>
> ⚠️ **This tree was recently refactored.** If your blog posts or old
> tutorials reference `aten/src/ATen/native/cuda/ConvolutionKernel.cu` or
> `ConvBiasEpilogueKernel.cu`, they describe the *pre-refactor* tree. The cuDNN
> conv code now lives in `aten/src/ATen/native/cudnn/` (`ConvShared.cpp`,
> `Conv_v7.cpp`, `Conv_v8.cpp`), and the autograd formulas have moved to
> `tools/autograd/derivatives.yaml`.

---

## TL;DR — the pipeline

```
F.conv2d(x, w, b)           x, w, b on device="cuda" (NCHW, fp16/fp32/bf16)
   │
   ▼  PyTorch core (same funnel as the TPU path)
aten::conv2d → conv2d_symint → convolution_symint → aten::convolution
   │
   ▼  DISPATCH DIVERGES HERE (device = "cuda")
_convolution (aten/src/ATen/native/Convolution.cpp)
   │  _select_conv_backend → ConvBackend::Cudnn
   │  memory format: NCHW or channels-last NHWC (heuristic)
   ▼
at::cudnn_convolution(...)        ← NO bias argument
   │  ConvShared.cpp: output alloc + raw dispatch
   │  runtime switch: v8 (default) / v7 (classic API, env toggle)
   ├─ v8: cuDNN Frontend/Backend API — op graph, engine plan cache,
   │      heuristic or exhaustive plan search → cudnnBackendExecute
   ├─ v7: classic API — descriptors + algo search
   │      (cudnnGet… / cudnnFind…) → cudnnConvolutionForward
   ▼  cuDNN (CLOSED SOURCE) picks a prebuilt kernel family
   │   (implicit GEMM / Winograd / FFT / direct, tensor-core variants)
   │   and launches it on the current CUDA stream
   ▼
output.add_(bias)                 ← SEPARATE in-place add kernel
result lives in device memory (possibly physically NHWC)
```

The single most important architectural fact: **there is no intermediate IR
and no graph compilation on this path.** Each conv call is one cuDNN call,
per-op, with an *algorithm* (not a program) chosen per shape and cached
in-process. cuDNN itself — the library that turns the call into actual
kernels — is closed source.

---

## Stage 0 — PyTorch core (no CUDA code)

Identical to the TPU path: `torch.nn.functional.conv2d` goes through
autograd, and the native conv kernels funnel into the generic ops. From
[`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml):

```yaml
- func: conv2d(Tensor input, Tensor weight, Tensor? bias=None, ...) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv2d_symint        # L1719

- func: convolution(Tensor input, Tensor weight, Tensor? bias, ...) -> Tensor
  dispatch:
    CompositeExplicitAutograd: convolution          # L1680

- func: convolution_backward(...) -> (Tensor, Tensor, Tensor)
  dispatch:
    CompositeExplicitAutograd, CUDA: convolution_backward   # L1686

- func: cudnn_convolution(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride,
                          SymInt[] dilation, SymInt groups, bool benchmark,
                          bool deterministic, bool allow_tf32) -> Tensor
  dispatch:
    CUDA: cudnn_convolution                         # L1885 — NOTE: no bias argument
```

Note the last entry: `cudnn_convolution` **does not take a bias**. That is a
deliberate design of the refactored tree (see [Quirks](#quirks-worth-knowingcudnnspecific)).

---

## Stage 1 — The ATen funnel (backend-agnostic, shared with CPU)

### 1a. `conv2d_symint`

[Convolution.cpp#L1038](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1038):
dtype check, `batchify`, complex-number path, then
→ `at::convolution_symint` (L1058).

### 1b. `_convolution` — the funnel and the backend switch

[`Convolution.cpp#L1576`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1576)
(`_convolution`, called from the `convolution` op):

1. Expand scalar params (`stride=2` → `[2, 2]`) — L1590–1602
2. `check_shape_forward` — L1604
3. 1-D → 2-D view for backends without native 1-D conv — L1612–1617
4. **`select_conv_backend`** ([L1297](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1297)) → `ConvBackend::Cudnn` for a CUDA input that passes the cuDNN gates (compiled with cuDNN, right dtype, not a double, etc.)
5. **`determine_backend_memory_format`** ([L1570](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1570)) → NCHW or **channels-last NHWC** (heuristic below)

Then the backend call (L1638–1646):

```cpp
case ConvBackend::Cudnn:
  check_input_same_type_as_parameters(input, weight, bias);
  output = at::cudnn_convolution(
      input.contiguous(backend_memory_format), weight, params.padding, params.stride,
      params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
  if (bias.defined()) {
    output.add_(reshape_bias(input.dim(), bias));   // ← bias: a SEPARATE in-place add
  }
  break;
```

Source: [Convolution.cpp#L1640](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1640)

**Channels-last heuristic** — [`ConvUtils.h#L348`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L348)
(`_conv_suggest_memory_format_impl`, via `cudnn_conv_suggest_memory_format`
at [ConvUtils.h#L370](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L370)):
if the *input or weight* is already channels-last (and dtype ≠ float64), run
the conv in `ChannelsLast` (NHWC); otherwise NCHW. This is why
`model.to(memory_format=torch.channels_last)` changes kernel selection without
any code change in your model.

Because `conv2d` is `CompositeImplicitAutograd`, its backward is derived by
differentiating this composite implementation (the modern autograd formulas
live in [`tools/autograd/derivatives.yaml`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)).

---

## Stage 2 — The `cudnn_convolution` ATen/CUDA wrapper

`aten/src/ATen/native/cudnn/` is the conv-specific cuDNN layer:

| File | Role |
|---|---|
| `ConvShared.cpp` | public ATen ops (`cudnn_convolution`, `…_backward_input`, `…_backward_weight`, `…_add_relu`), shape checks, output allocation |
| `Conv_v8.cpp` | **cuDNN v8-era path (default)** — Frontend C++ API + Backend execution, engine-plan cache, heuristic/exhaustive plan search |
| `Conv_v7.cpp` | **cuDNN classic ("v7") API path** — descriptors + `cudnnGet/Find…Algorithm` + `cudnnConvolutionForward/Backward*` |
| `ConvPlaceholders.cpp` | stubs when ATen is built without cuDNN |

### 2a. `cudnn_convolution`

[ConvShared.cpp#L243](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L243):

```cpp
Tensor cudnn_convolution(const Tensor& input_t, const Tensor& weight_t, ...) {
  auto memory_format = cudnn_conv_suggest_memory_format(input_t, weight_t);  // L255
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(input_t.sizes(), weight_t.sizes(), padding, stride, dilation),
      input->options().memory_format(memory_format));                       // L256-259
  ...
  cudnn_convolution_forward_out(output, c, input, weight, ...);            // → raw dispatch
}
```

→ `cudnn_convolution_forward_out` ([ConvShared.cpp#L212](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L212))
(type/GPU checks, `contiguous(memory_format)`) →
**`raw_cudnn_convolution_forward_out`** — whose implementation differs per
API era and is picked **at runtime**:

```cpp
// Conv_v8.cpp:1464
void raw_cudnn_convolution_forward_out(...) {
  ...
  if (at::native::cudnnv8_enabled_check_debug()) {          // L1481 — default: true
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, ...);  // v8
  } else {
    raw_cudnn_convolution_forward_out_v7(...);              // L1495 — classic API
  }
}
```

The gate: [`ConvUtils.h#L96`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L96)
— `cudnnv8_enabled_check_debug()` reads env **`TORCH_CUDNN_V8_API_DISABLED`**
(v8 on by default; set the env var to fall back to the classic API).
Rationale comment at
[ConvShared.h#L139](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.h#L139):
"v7 functions are preserved here to allow for runtime switching to v7 … v7
forward/backward out can have different behavior from the v8 versions, as v7
explicitly splits large tensors as a 32-bit indexing workaround whereas v8
expects cuDNN to handle large tensors."

---

## Stage 3 — The cuDNN call

### 3a. v8 path (default): Frontend API, engine plans

`run_single_conv` ([Conv_v8.cpp#L1152](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1152)):

1. **Handle** — `getCudnnHandle()` (L1164). The handle is created in
   [`aten/src/ATen/cudnn/Handle.cpp#L11`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L11)
   (`cudnnCreate`) and bound to the *current* CUDA stream at each use:
   [`Handle.cpp#L53`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L53)
   (`cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream())`).
2. **Plan cache** — a `CacheKeyWrapper` over
   (operation, output, input, weight, padding, stride, dilation, groups,
   deterministic, allow_tf32) (L1165–1175); `_get_benchmark_cache()->find(key)`
   (L1177) → `run_conv_plan` (L1180). A hit replays the cached engine plan.
3. **On a miss, without benchmark** — engine configs from cuDNN
   heuristics, tried in order: top config (L1189), all heuristic configs
   (L1219), fallback configs (L1254); `try_configs` builds the plan for each
   candidate and executes it (workspace comes from PyTorch's caching
   allocator).
4. **On a miss, with benchmark** (`torch.backends.cudnn.benchmark = True`) —
   `get_plans_from_find` (L1282): an *exhaustive* search over cuDNN engine
   plans, then `try_plans` over all of them.
5. The op graph for this path is a **single conv op**
   (`CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`, L1483) built
   with the cuDNN **Frontend C++ API** (`cudnn_frontend::Tensor`,
   `TensorBuilder` — see `getTensorDescriptorWithTypeVirtual` at
   [Conv_v8.cpp#L60](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L60)); plan lookup goes through
   `cudnnGetPlan` (L771); execution is `cudnnBackendExecute`.

**Fused variant** — `raw_cudnn_convolution_add_relu_out`
([Conv_v8.cpp#L1593](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1593))
builds a **single cuDNN graph of conv + bias + add + ReLU**
(`get_plans_from_find_fused` L865, `try_plans_fused` L1447). It is *not* used
by eager `F.conv2d`; it backs the ops `cudnn_convolution_relu` /
`cudnn_convolution_add_relu` (yaml L1908/L1913), which the **TorchScript
frozen-model fusion pass**
([`frozen_conv_add_relu_fusion_cuda.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp#L39))
substitutes for `conv + add + relu` patterns in frozen/quantized models.

### 3b. v7 path (classic API): descriptors, algo search, `cudnnConvolutionForward`

`raw_cudnn_convolution_forward_out_v7`
([Conv_v7.cpp#L840](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L840)):

1. **32-bit workaround** — `split_batch_dim_to_32bit_out`
   ([Conv_v7.cpp#L673](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L673)):
   if input/output numel exceeds int32, the batch dim is explicitly split and
   the conv is run in slices (the v8 path instead expects cuDNN to handle
   large tensors).
2. **Descriptors** (`raw_cudnn_convolution_forward_out_32bit`,
   [Conv_v7.cpp#L757](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L757), L770–795):

   | Descriptor | What goes in | Source |
   |---|---|---|
   | `TensorDescriptor` (input/output) | real strides → NCHW **or NHWC** via `cudnnSetTensorNdDescriptor` | [`Descriptors.h#L169`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L169) |
   | `FilterDescriptor` (weight) | `cudnnSetFilterNdDescriptor` | [`Descriptors.h#L191`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L191) |
   | `ConvolutionDescriptor` | `cudnnSetConvolutionNdDescriptor(dim, pad, stride, dilation, CUDNN_CROSS_CORRELATION, mathType)` + `cudnnSetConvolutionGroupCount(groups)` + initial math type | [`Descriptors.h#L203`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L203) |

   Two details that matter:
   - **`CUDNN_CROSS_CORRELATION`** (Descriptors.h:207): cuDNN's default is a
     true convolution (flipped kernel); PyTorch's conv is cross-correlation,
     so the flag is set explicitly.
   - **Initial math mode** (Descriptors.h:210–215): fp16 →
     `CUDNN_TENSOR_OP_MATH` (tensor cores); fp32 without `allow_tf32` →
     `CUDNN_FMA_MATH` (forbids TF32); otherwise `CUDNN_DEFAULT_MATH`.
3. **Algorithm selection** — `AlgoIterator::try_all`
   ([Conv_v7.cpp#L582](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L582)):
   - `deterministic && !benchmark` → `DEFAULT_ALGO` only
     (L566–576, with the fp16/TF32 math-mode rules above)
   - otherwise: **`BenchmarkCache`** lookup (L105–132, keyed by
     `ConvolutionParams` — shapes + params + benchmark/deterministic/TF32)
   - on miss: `findAlgorithms` → heuristic
     `cudnnGetConvolutionForwardAlgorithm_v7` (L304) or exhaustive
     `cudnnFindConvolutionForwardAlgorithmEx` (L319, when
     `benchmark=true`); benchmarking clears the CUDA allocator cache
     afterwards (L528)
   - each candidate algo is tried; OOM/cuDNN error → next candidate; a
     fallback search runs if the first list fails (L622–631)
4. **Execution** (L802–834): allocate the algo's workspace from the caching
   allocator (L804), re-set `cudnnSetConvolutionMathType` to the chosen
   mathType (L811 — the "Note [behavior of cudnnFind and cudnnGet]" at the
   top of the file explains why), then:

   ```cpp
   cudnnConvolutionForward(handle, &one /*alpha=1*/,
       idesc.desc(), input.const_data_ptr(),
       wdesc.desc(), weight.const_data_ptr(),
       cdesc.desc(), fwdAlgPerf.algo,
       workspace.data_ptr(), fwdAlgPerf.memory,
       &zero /*beta=0*/,
       odesc.desc(), output.data_ptr());        // Conv_v7.cpp:819
   ```

   Source: [Conv_v7.cpp#L819](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L819)

---

## Stage 4 — Inside cuDNN (closed source)

⚠️ **This stage is closed source** — it lives in NVIDIA's `libcudnn`
(+ `libcudnn_cnn_infer`, etc.). PyTorch hands cuDNN the descriptors (or a
v8 op graph) and an algorithm/plan; what happens next is opaque:

- **Algorithm families**: implicit GEMM (the workhorse — the conv is
  internally an im2col-style GEMM on tensor cores), Winograd (FFT-based
  transforms), FFT, direct. Each has tensor-core / FP16 / BF16 variants.
- **Kernel selection, tiling, and parallelization** for the chosen family —
  per SM architecture (the kernels are prebuilt; a small number of ops may
  JIT via NVRTC).
- The **workspace** semantics (size, layout) are cuDNN's; PyTorch just
  allocates it.

You can only observe its *output*:

- **Nsight Systems** kernel names encode the family, e.g.
  `sm80_xmma_tensorop_f16_convndf_an_f0n0c32k32r3s3…` (implicit-GEMM
  forward), `…wgrad…` / `…dgrad…` for backward filter/data — see
  [Quirks](#quirks-worth-knowingcudnnspecific) item 7.
- **`CUDNN_LOGLEVEL_DEBUG` + `CUDNN_LOGDEST_DBG`** (NVIDIA-documented,
  cuDNN User Guide): the debug log prints the graphs cuDNN sees, the
  candidate algorithms and their measured timings when benchmarking.
- The v7 heuristic/benchmark results (which algo won, its workspace size)
  are logged by PyTorch's error paths and `TORCH_CUDNN_V8_API_DEBUG=1`.

---

## Stage 5 — Launch and result

- cuDNN launches its kernel(s) **on the handle's stream = the current CUDA
  stream** ([Handle.cpp#L53](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L53)) — asynchronous; the CPU keeps going.
- The output tensor is a device tensor in the *suggested* memory format —
  **it may be physically NHWC** (channels-last strides) even though your code
  is NCHW. A later `.contiguous()` would then pay a transpose kernel.
- The bias `add_` (Stage 1) launches a second, separate elementwise kernel.
- `.cpu()` = a device→host copy through the caching allocator.

So a single eager `F.conv2d` with bias, end to end, is **two device kernels**:
the cuDNN conv kernel + the bias-add kernel.

---

## Backward

Autograd formula for `cudnn_convolution` —
[`tools/autograd/derivatives.yaml#L2769`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml#L2769):

```yaml
- name: cudnn_convolution(Tensor self, Tensor weight, ...) -> Tensor
  self, weight: "_cudnn_convolution_backward(self, grad, weight, padding, ...,
               {grad_input_mask[0], grad_input_mask[1]})"
```

→ `_cudnn_convolution_backward`
([FunctionsManual.cpp#L7590](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp#L7590)).
The comment above it explains the design:

> "This function only exists because **cuDNN does not support bias gradient
> computation** and it's not easy to slice a std::tuple to return only
> grad_input / grad_weight from `convolution_backward`."

It calls the general backward with the bias gradient masked out:
`at::convolution_backward_symint(grad_output, self, weight, nullopt, …,
{mask_in, mask_w, false})`.

`convolution_backward`
([Convolution.cpp#L2086](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L2086),
yaml L1686 `CUDA: convolution_backward`) then selects `ConvBackend::Cudnn`
(L2177) and calls `cudnn_convolution_backward_stub` (L2181) →
`cudnn_convolution_backward` ([ConvShared.cpp#L584](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L584)),
which computes each gradient *only if requested* by `output_mask`:

| Gradient | ATen entry | v8 (default) | v7 (classic) |
|---|---|---|---|
| grad_input | `cudnn_convolution_backward_input` (ConvShared.cpp:368) | `raw_…_backward_input_out` (Conv_v8.cpp:1509) → `run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)` | `cudnnConvolutionBackwardData` ([Conv_v7.cpp#L930](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L930)) |
| grad_weight | `cudnn_convolution_backward_weight` (ConvShared.cpp:505) | `raw_…_backward_weight_out` (Conv_v8.cpp:1551) → `…_BACKWARD_FILTER_DESCRIPTOR` | `cudnnConvolutionBackwardFilter` ([Conv_v7.cpp#L1049](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L1049)) |
| grad_bias | — (not requested by cuDNN path) | — | — |

**grad_bias comes from the separate `add_`** (Stage 1): autograd's backward
for the in-place add is a sum over batch + spatial dims — a plain CUDA
sum-reduce kernel. (A generic fallback for backends that don't return a bias
gradient lives at
[Convolution.cpp#L2380](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L2380):
`grad_output.sum(IntArrayRef{0, 2, 3})`.)

**Contrast with TorchTPU**: the CUDA path does *not* use the old
"role-swap" trick (computing grad_weight as a forward conv with input/weight
exchanged). cuDNN provides *dedicated* bwd-data and bwd-filter calls — the
same adjoint mathematics, pre-implemented by NVIDIA as kernels. TorchTPU,
having no such kernels in its IR, instead emits adjoint
`stablehlo.convolution`s with flipped dimension numbers.
([Comparison doc](cuda-vs-tpu-conv2d-comparison.md).)

**Non-cuDNN fallbacks** (when the cuDNN gates fail):
`ConvBackend::CudaDepthwise2d/3d` (custom depthwise kernels,
`aten/src/ATen/native/cuda/DepthwiseConv2d.cu`) and the "slow" paths —
which on CUDA are *im2col + cuBLAS GEMM*, not naive loops:
[`aten/src/ATen/native/cuda/ConvolutionMM2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu)
(`at::native::im2col` at L182/L338, `at::cuda::blas::gemm` at L203/L359;
`_slow_conv2d_backward` at L216).

---

## Observing every stage yourself

```python
import torch
# knobs (all settable in code; env-var equivalents exist):
torch.backends.cudnn.benchmark   = True    # exhaustive algo search (first call per shape is slower)
torch.backends.cudnn.deterministic = True  # force cuDNN DEFAULT_ALGO
torch.backends.cudnn.allow_tf32  = False   # fp32: forbid TF32 (CUDNN_FMA_MATH)

x = torch.randn(4, 8, 32, 32, device="cuda", dtype=torch.float16)
w = torch.randn(16, 8, 3, 3,  device="cuda", dtype=torch.float16)
b = torch.randn(16,           device="cuda", dtype=torch.float16)
y = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
```

What to look at:

- **torch profiler** (`torch.profiler.profile`) — you'll see
  `aten::cudnn_convolution` **and** `aten::add_` as separate nodes, and two
  device kernels for the biased conv (the conv kernel + a `vectorized_elementwise_kernel` for the bias).
- **Nsight Systems** (`nsys profile python train.py`) — the actual kernel
  names. The implicit-GEMM family reads e.g.
  `sm80_xmma_tensorop_f16_convndf_an_f0n0c32k32r3s3t3x1w1u1…`; Winograd and
  direct families have distinct names; backward shows `wgrad`/`dgrad`
  variants.
- **cuDNN debug log** (NVIDIA-documented env vars):
  ```bash
  CUDNN_LOGLEVEL_DEBUG=4 CUDNN_LOGDEST_DBG=/tmp/cudnn.log python train.py
  ```
  → the graph cuDNN sees, candidate algos, and their measured times when
  benchmarking. This is the closest thing to an "HLO dump" on the CUDA side.
- **Force the classic API** to watch the v7 path:
  `TORCH_CUDNN_V8_API_DISABLED=1` (add `TORCH_CUDNN_V8_API_DEBUG=1` to see
  the gate log).
- **Watch the layout decision**:
  `model.to(memory_format=torch.channels_last)` → the same logical NCHW conv
  now runs on NHWC data (often a visible speedup on tensor-core GPUs).

---

## Quirks worth knowing (cuDNN-specific)

1. **The bias is a separate `add_` kernel in eager mode.** The v8 single-op
   graph contains no bias node, and `cudnn_convolution` has no bias
   argument. (The *frozen-model* path — TorchScript `frozen_conv_add_relu`
   fusion — is where conv+bias+relu collapse into one cuDNN graph op.)
2. **The output can be physically NHWC** even when your code says NCHW —
   channels-last is suggested whenever the input *or* weight is
   channels-last (ConvUtils.h:348).
3. **Two cuDNN API generations coexist at runtime**: v8 (Frontend/Backend
   API, engine-plan cache) is the default; the classic v7 API is retained and
   toggled by `TORCH_CUDNN_V8_API_DISABLED=1` (ConvShared.h:139).
4. **Algo caching is in-process and per-shape** — `BenchmarkCache` keyed by
   `ConvolutionParams` (Conv_v7.cpp:105); it does not survive process exit
   (contrast TorchTPU's 3-tier compile cache).
5. **`CUDNN_CROSS_CORRELATION`** — PyTorch conv is cross-correlation; cuDNN's
   default is a flipped-kernel convolution (Descriptors.h:207).
6. **Precision math modes**: fp16/bf16 → tensor cores
   (`CUDNN_TENSOR_OP_MATH`); fp32 → TF32 is opt-in via
   `allow_tf32`/`CUDNN_FMA_MATH` (Descriptors.h:210–215, Conv_v7.cpp:566–576).
7. **Kernel names are your window into the closed stage** — the Nsight
   naming (`smXX_xmma_tensorop_…_convndf_…` vs Winograd/direct) tells you
   which algo family cuDNN chose without any source.
8. **`deterministic && !benchmark` → `DEFAULT_ALGO` only** (Conv_v7.cpp:583)
   — determinism costs performance; the same trade as TorchTPU's
   precision-in-cache-key, but coarser.
9. **cuDNN computes no bias gradient** — hence `_cudnn_convolution_backward`
   existing at all (FunctionsManual.cpp:7585–7589); grad_bias is a generic
   sum.
10. **Large tensors**: the v7 path explicitly splits the batch for 32-bit
    indexing (Conv_v7.cpp:673); v8 delegates to cuDNN (ConvShared.h:139–146).

---

## Source index

### `pytorch/pytorch` (branch `main`, commit `cb9eb4b`)

| Concern | File | Key lines |
|---|---|---|
| Op declarations (`conv2d`, `convolution`, `cudnn_convolution`) | [`aten/src/ATen/native/native_functions.yaml`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) | [L1719](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1719), [L1680](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1680), [L1686](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1686), [L1885](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L1885) |
| `conv2d_symint` | [`aten/src/ATen/native/Convolution.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp) | [L1038](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1038) |
| `_convolution` funnel + backend switch + bias `add_` | same | [L1576](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1576), [L1297](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1297), [L1638–1646](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L1638) |
| `convolution_backward` (backend switch, bias-sum fallback) | same | [L2086](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L2086), [L2177–2188](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L2177), [L2380](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp#L2380) |
| channels-last heuristic | [`aten/src/ATen/native/ConvUtils.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h) | [L348](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L348), [L370](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L370), [L96](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ConvUtils.h#L96) (v8 gate) |
| `cudnn_convolution` / `…_backward_input` / `…_backward_weight` | [`aten/src/ATen/native/cudnn/ConvShared.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp) | [L243](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L243), [L368](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L368), [L505](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L505), [L584](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L584), [L710](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.cpp#L710) (`…_add_relu`) |
| v7/v8 layering + large-tensor note | [`aten/src/ATen/native/cudnn/ConvShared.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.h) | [L139](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/ConvShared.h#L139) |
| v8 path: `run_single_conv`, plan cache, heuristics, find, fused plans | [`aten/src/ATen/native/cudnn/Conv_v8.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp) | [L1152](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1152), [L1177](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1177), [L1189](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1189), [L1282](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1282), [L1464](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1464), [L1483](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1483), [L1593](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1593) |
| v7 path: 32-bit split, descriptors, algo search, forward/bwd calls | [`aten/src/ATen/native/cudnn/Conv_v7.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp) | [L105](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L105), [L304](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L304), [L319](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L319), [L566](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L566), [L582](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L582), [L673](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L673), [L757](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L757), [L819](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L819), [L930](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L930), [L1049](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v7.cpp#L1049) |
| cuDNN descriptors (NCHW/NHWC strides, cross-correlation, math modes) | [`aten/src/ATen/cudnn/Descriptors.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h) | [L169](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L169), [L191](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L191), [L203](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Descriptors.h#L203) |
| cuDNN handle + stream binding | [`aten/src/ATen/cudnn/Handle.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp) | [L11](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L11), [L53](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cudnn/Handle.cpp#L53) |
| cuDNN backward autograd formula | [`tools/autograd/derivatives.yaml`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml) | [L2769](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml#L2769) |
| `_cudnn_convolution_backward` (+ "no bias gradient" comment) | [`torch/csrc/autograd/FunctionsManual.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp) | [L7590](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp#L7590) |
| Frozen-model conv+add+relu fusion (uses fused cuDNN graph ops) | [`torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp) | [L39](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp#L39), [L58](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/passes/frozen_conv_add_relu_fusion_cuda.cpp#L58) |
| im2col + GEMM fallback ("slow" CUDA conv) | [`aten/src/ATen/native/cuda/ConvolutionMM2d.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu) | [L182](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu#L182), [L203](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu#L203), [L216](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu#L216), [L359](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ConvolutionMM2d.cu#L359) |

### Closed-source stage (for reference, not linked)

- **NVIDIA cuDNN** (`libcudnn`, `libcudnn_cnn_infer`, …) — algorithm
  families (implicit GEMM / Winograd / FFT / direct), kernel selection and
  codegen/tiling, fused graph execution. Observable via Nsight kernel names
  and `CUDNN_LOGLEVEL_DEBUG`/`CUDNN_LOGDEST_DBG` (cuDNN User Guide,
  "Profiling and Debugging").

---

*Caveats: (1) line numbers were correct at commit `cb9eb4b` (2026-11-01) and
will drift; symbol names and file paths are the durable anchors — and note
the `native/cuda/ConvolutionKernel.cu` → `native/cudnn/` move means older
references are stale. (2) Stage 4 is closed source by design — every claim
in that section is about what is (or is not) publicly observable, not about
NVIDIA's internals. (3) This document covers the **native eager** path;
`torch.compile` on CUDA (Triton/cuDNN-graphs) is a different pipeline.*
