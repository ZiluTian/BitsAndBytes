# Fusing the Q/DQ chain, and other bit-exact compiler-side co-design wins

> Implementation notes on making a quantized reference model's forward fast **without changing a single
> result** — for both the mixed-precision Tuner that evaluates it hundreds of times and the on-target (NPU)
> compiler that consumes it. The centerpiece is a fused **quantize/dequantize (Q/DQ)** Triton kernel; the rest
> is a set of bit-exact co-design changes that kill the kernel-launch and host↔device sync storms. Every change
> is result-bit-exact and stays hardware-legal. The implementation half of
> ["make the tuner faster"](making-the-tuner-faster.md) — those "same answer, faster" wins, at the compiler level.
> Measured 2026-10, A/B/A on the same host.

---

## TL;DR

A small, op-heavy quantized reference model is not matmul-bound. It's **kernel-launch-bound and
sync-bound**: every operator is `quantize → core → dequantize`, and the Q/DQ halves are each a short chain of
small elementwise kernels. On one measured forward, the Q/DQ chain was **~85% of the compute kernels** and the
forward ran **~2× longer than the GPU work itself** (CPU dispatch-bound). The fix is to co-design the *representation
the NPU consumes* — how Q/DQ is expressed, how calibration stats flow, how quantizer params are stored — so the
whole stack is leaner, **while the output stays bit-identical**.

| Change | What it removes | Measured |
|---|---|---|
| **Fused Q/DQ Triton kernel** | ~2,600 elementwise launches/forward | eval pass 2.42 s → 1.13 s (2.14×); full run −27% |
| **On-device calibration min/max** | ~79k host↔device syncs per calibration pass | calib 9.8 s → 2.8 s (0 DtoH) |
| **Cached ReLU6 bound** | ~72 DtoH + 34 HtoD syncs/forward | DtoH 5,690 → 318/pass; CPU −21%; GPU mean util 22% → 89% |
| **Dead-buffer removal + BN fast path** | ~204 clones + DtoD/forward; per-call BN recompute | eval pass −25%; full run −20% |
| **Direct eval return (no clone/copy_)** | 736 DtoD memcpys/forward | CUDA events 4,166 → 3,430; forward −17.6% |
| **Quantizer params as host scalars** | 0-dim tensor qparam handling | bit-exact; unblocks the on-target export scalar path |

The one word that governs all of it: **bit-exact**. A speedup that changes the answer is a regression.

---

## The frame: what the NPU compiles, and why the forward is the bottleneck

The NPU compiles a **quantized reference model**. Every operator in it is a three-step sandwich:
`quantize(input) → core_op → dequantize(output)`. For integer quantization the "quantize" is not one op — it's a
short elementwise chain (scale, add zero-point, round, clamp, subtract zero-point, rescale). Multiply that by
every quantizer in every layer, and a small model has hundreds of Q/DQ pairs per forward.

I profiled one eager forward of such a model. The budget:

| Component | Count | Note |
|---|---|---|
| `nn.Module.__call__` invocations | 618 | |
| CUDA device events / forward | 4,166 | 11.2 ms of device time |
| — compute kernels | 3,083 | 8.9 ms |
| — — of which the **Q/DQ elementwise chain** | **~2,600** | **≥85% of compute kernels** |
| — — of which conv / pool / BN | ~400 | the "real" ops |
| — eval `clone` + `copy_` DtoD memcpys | 736 | most of the 2.4 ms memcpy time |
| — other memcpys + memsets | ~347 | |
| **CPU aten dispatch / launch** | **~26 ms** | **vs ~11 ms device work** |

The punchline: **~26 ms of CPU dispatch against ~11 ms of GPU work** — the forward is CPU/launch-bound ~2×, and
the Q/DQ chain (thousands of tiny elementwise launches) plus a few hundred DtoD copies are the dominant CPU
cost. There's nothing here to "scale up"; there's a representation problem to fix. That's the whole game: make the
model the NPU compiles (and the Tuner evaluates) leaner, **without moving a bit of the result**.

---

## 1) Fuse the Q/DQ chain into one Triton kernel

**Problem.** Each Q/DQ pair was ~7 small elementwise kernels (scale, +zero-point, `floor(x+0.5)`, clamp,
−zero-point, /scale). Hundreds of pairs → thousands of kernels → thousands of CPU dispatches, and the GPU sits
between them.

**Design.** One elementwise Triton kernel per pair, computing both halves:

```
q = clamp( floor(x * scale + 0.5), qmin, qmax )   # quantize
y = q / scale                                      # dequantize
```

**Why it's bit-exact *by construction* — not just "approximately equal."** The scale `2**(qlen − qm)` is a power
of two, which is **exact in float32**. That means the *only* rounding anywhere in the whole chain is the single
`+0.5` before `floor` (round-half-up). The eager chain and the fused kernel perform the same arithmetic in the
same order, so the result is identical — no tolerance, no "within epsilon." (This is why you must never reach for
`torch.round` / round-half-to-even here; it would change the bits.) There's also **no host↔device sync** in the
kernel: its arguments (scale, qmin, qmax) are computed host-side from the integer quantizer format.

**Safe by design.** It's **opt-in and OFF by default** — zero behavior change unless enabled — and it falls back
to the eager path for anything off the plain formula: non-CUDA / non-f32 / non-contiguous inputs, missing Triton,
or a quantizer that isn't the plain formula (e.g. a 4-bit codebook quantizer).

**Measured.** Validation pass **2.42 s → 1.13 s (2.14×)**; a full tuning run **~27% faster** (285.7 s → 208.4 s).
Bit-exactness was checked three ways: a grid of quantizer configs × shapes all `torch.equal` vs the eager path;
the full validation set's loss and accuracy unchanged to the last bit; and the full tuning run produces an
**identical tuned model**.

---

## 2) Keep calibration stats on-device (kill the sync storm)

**Problem.** In calibration mode, every quantizer round-tripped its running min/max to the host **per batch** —
with on the order of 1,500 quantizers that's tens of thousands of DtoH syncs in a single calibration pass
(~79k measured). Each one drains the GPU pipeline, exactly like a per-batch `.item()`.

**Design.** Accumulate the running min/max **on the input device** (no round-trip per batch), and **finalize the
quantizer format exactly once** when the stats are first consumed (at export / at the calibrate→eval transition):

```
per batch, per quantizer:   stats.min = min(stats.min, input.min())   # on-device
                           stats.max = max(stats.max, input.max())    # on-device
finalize:                    qformat = resolve(stats)                # once
```

**Why it's bit-exact.** Min and max are **exact, order-independent reductions** — the merged/finalized value is
identical to the serial per-batch one, so the resolved quantizer format is unchanged. This same property is what
makes **multi-GPU sharded calibration** bit-identical too: each device calibrates a disjoint batch shard and the
per-quantizer min/max merge back exactly.

**Measured.** A single calibration pass: **9.8 s → 2.8 s with 0 DtoH syncs** (3.5× over the original, 12.3× over
the pre-fix state). Every reported result across the bit-width grid matches the per-batch-sync baseline exactly.

---

## 3) Cache the ReLU6 quantized bound host-side

**Problem.** A ReLU6 layer rebuilt its saturation bound on every call: a small pageable host→device copy, quantize
it to a 0-dim device tensor, then the hard-tanh guard compares a float to that tensor — which **syncs to fetch the
value**. Two pinned DtoH scalar fetches plus one HtoD **per ReLU6 layer, per forward**. With a couple of dozen ReLU6
layers that's hundreds of syncs a forward.

**Design.** The quantized bound is a **pure function of the activation quantizer's host-side state**, so cache it
host-side as a Python float, keyed on that full state (bit-width, offset, sign, calibrated min/max, eval mode):

```
bound = cache[state] or rebuild(state)     # a host float; rebuilt on any state change
hardtanh(x, 0.0, bound)                    # scalar-bound clamp path — no device comparison
```

The cache rebuilds automatically whenever any part of the state changes in place (calibration commits min/max, the
calibrate↔eval flag flips, quantizers are re-created, `step()` runs) — so it can never go stale.

**Measured.** DtoH-pinned events **5,690 → 318 per pass** (72 → 4 per forward); CPU self time **−21%**; and on a
full tuning run GPU utilization went from a CPU-bound **mean 22% to mean ~89%** (the GPU is now the ceiling).
Bit-exact: a 30-check unit test (cold/warm, after `step()`, after an in-place format change, after deepcopy and
pickle) all `torch.equal` vs the per-call re-derivation, plus the full tuning flow bit-identical.

---

## 4) Stop doing dead work: dead-buffer removal + BN fast path

Two changes that stop paying for work whose result nobody needs:

- **Dead output buffer.** Every forward `detach().clone()`d every layer's output into a dictionary that **nothing
  in the framework ever read** — ~204 clones + ~204 DtoD copies + retained GPU memory per forward, and it also
  inflated the per-iteration `deepcopy` the Tuner does. Now gated off by default.
- **BatchNorm eval fast path.** In eval mode the per-channel BN coefficients are constant, but the forward
  recomputed them (`3× .to(device)` + `5× view().expand` + `sqrt(running_var + eps)`) per layer per forward. They're
  now computed once per module and reused; the **element-wise values and op order are unchanged**, so the output is
  bit-identical. (Careful here: invalidation must be explicit, because an in-place `.data.copy_` does *not* bump a
  tensor's version counter.)

**Measured.** Calibrated validation pass **~25% faster**; full tuning run **~20% faster**. Bit-exact: per-module
`torch.equal` vs the original formulation (cold/warm, after `step()`, after a training forward, after load and a
pickle round-trip, across all BN/standardizer modules) with the full validation loss/accuracy unchanged.

---

## 5) Return the eval result directly (drop the training-only clone/copy_)

**Problem.** The quantizer's fake-quantize step implements the straight-through estimator with a
`clone(input)` + in-place `copy_(quantized)` — which matters for **backward**, but runs in **both** training and
eval. In eval, the quantized value is already materialized in a fresh tensor; the clone+copy is pure waste (and was
the 736 DtoD memcpys/forward in the budget above).

**Design.** In eval, return the already-quantized tensor directly; the training path is left byte-identical.

**Measured.** CUDA device events **4,166 → 3,430** — *exactly* the 736 DtoD memcpys and nothing else (verified by
event-level diff); forward wall **−17.6%**; full validation pass **−16.5%**. Bit-exact by construction: the
training path is untouched, so QAT bit-exactness is preserved.

---

## 6) Quantizer params as host scalars

Small, but it mattered in a place I didn't expect. Quantizer params (scale / zero-point) were 0-dim tensors; making
them plain host floats is bit-exact (a power-of-two scale is exact in f32) and removes a class of tiny copies for
in-process models. The surprise: it turned out to be **required for the on-target export** — exporting a
Python-float scalar through the quantize path had been doing pure-Python arithmetic that produced a float the
integer ops reject. Normalizing non-tensor inputs (`as_tensor`) at the base quantize/dequantize restored the
exact 0-dim behavior for scalars while leaving the tensor hot path a no-op. **Bit-exact, and it unblocked the export.**

---

### (And on the eval side, briefly)

The forward wasn't the only CPU-bound piece — the validation/eval loop had the same shape of bug: per-image
decode on the main thread, a per-batch `.item()` drain, and a data loader that read a `num_workers` setting it
never passed through. Preloading the validation set into RAM, accumulating the loss **on the GPU with one `.item()`
per pass**, and actually passing `num_workers`/`pin_memory` are the same "don't sync per batch, don't decode on
the critical path" principle applied to the harness. Same bit-exact discipline; they belong to the validator, not
the compiler, so they're a footnote here.

---

## The invariant: bit-exact and hardware-legal

Every change above obeys the same contract, because the model both feeds the Tuner's decisions and is what the NPU
compiles — a changed bit is a changed answer:

- **Rounding is `floor(x + 0.5)`** — never round-half-to-even; re-association of the Q/DQ arithmetic is forbidden.
- **float32, unchanged op order** — no fp16/autocast anywhere in the optimized path.
- **The training path is byte-identical** — every eval-only shortcut sits behind a mode/training gate.
- **Stale-cache protection** — any device-side cache is invalidated when the quantizer state it derives from
  changes (a set offset must take effect; a mode flip must re-derive).
- **Proof, not assertion** — per-batch losses bit-identical; full validation loss/accuracy unchanged; and a full
  tuning run ends on a **bit-identical tuned model**.

**Considered and rejected:** fp16/autocast (breaks bit-exactness); `cudnn.benchmark` / channels-last (measured no
effect on this model). Bigger validation batches and `torch.compile` are available as opt-in levers but change the
reported (batch-weighted) loss, so they're kept out of the deterministic default.

---

## Why this is co-design, not just "make it faster"

None of this is "speed up one kernel." It's **redesigning the representation the NPU consumes** — how a
quantized operator is expressed (one fused elementwise op instead of seven), how calibration state flows (on-device
until a single finalize), and how quantizer parameters are stored (host scalars) — so that *the model the compiler
compiles and the Tuner evaluates is simultaneously leaner and bit-identical*. The gains compound: the Tuner
evaluates this forward hundreds of times, so a 2× per-eval win is a 2× win on the whole search; and because every
change is result-bit-exact and hardware-legal, the NPU's outputs don't change at all. You get the speed for free,
against the hardware contract.

That's the co-design part: the speedup isn't a property of one component — it falls out of the *interface* between
the reference model and the compiler being cleaner, with the bit-exact invariant held across the whole stack.

---

## How the numbers were measured

- **A/B/A on the same host**, clean and under matched background load; a fresh process per mode; full
  validation passes and full tuning runs (not micro-benchmarks).
- **Attribution by event-level CUDA profiling** — the per-forward budget and each change's delta are read off the
  device-event / memcpy / sync counts, so a "2.14×" is tied to a specific removed kernel class, not a vibe.
- **Bit-exactness proven, not assumed** — `torch.equal` on quantizer configs × shapes, full validation-set
  loss/accuracy unchanged, and end-to-end tuned-model identity.
- Model and stack identifiers are withheld; the workload is a small, op-heavy quantized reference model with a few
  hundred Q/DQ pairs per forward.
