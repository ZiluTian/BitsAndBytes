# When the customer says "make the tuner faster"

> A field note on taking a mixed-precision PTQ tuner from *"too slow"* to *provably* faster — on a real
> customer detector, with the final model **bit-identical** to the slow run. No accuracy was traded; and a
> faster-but-riskier search I **chose not to ship**. All figures are wall-clock A/B measurements against the
> serial baseline, 2026-10.

---

## TL;DR

- A customer said, effectively, *"make it faster."* That's the whole brief.
- I took a **2 h 15 m** mixed-precision tuning run down to **16 m 09 s** — **8.4×** — with the final model
  **bit-identical** to the slow run (same configuration, same AP, same decisions).
- The clean, apples-to-apples number (parallel vs serial, *same* code): **3.18×**. Per tuning iteration:
  **14.6×** (117 s → 8 s median).
- I also built a search that ran **~5.4× faster** than the serial baseline and explored a *larger* space —
  and **deliberately didn't ship it** as the default, because measured against the metric the customer lives
  by it landed just under the accuracy floor.

The shape of the post: vague ask → diagnosis → scoping → a provable win → a judgment call on what *not* to ship.

---

## The ask

It arrived as a short line:

> "The tuner's too slow for our workflow. Can you make it faster?"

No target number. No deadline. No definition of "faster."

If you've done this work before you know the trap: read "faster" as "shave a constant somewhere" and hand back
a 5% win that doesn't move the customer's day. I didn't. I treated it like any field problem — **find out what
is actually slow, and find out what "faster" is allowed to mean** — before touching a line of code.

## What the tuner does

For context: our **mixed-precision Tuner** searches, per layer, over which quantizers stay at high precision
(16-bit) and which drop to low precision (8-bit), under an **accuracy-acceptance floor**, to minimize memory
for on-target (NPU) deployment. Two kinds of quantizers move per layer — **weights** and **output feature
maps** — and the search walks a ladder of *"how many of the least-sensitive layers can I drop to 8-bit while
still clearing the accuracy floor?"*

The customer's model was an **SSD/FPN-style object detector** — small 192×240 input, a few thousand validation
images, roughly a million detection boxes per evaluation. The baseline: a serial single-GPU run of the full
tuning task took **2 h 15 m**.

## First: what is this thing actually slow at?

I profiled one full evaluation of the tuner's decision loop. The result was the whole story:

| Per-evaluation stage | Time |
|---|---|
| NMS — pure-Python loop | 16.4 s |
| Per-image postprocess (Python) | 19.9 s |
| Prediction JSON write + re-parse (~170 MB) | 11.0 s |
| Ground-truth reload | 1.5 s |
| COCO matcher | 3.0 s |
| **GPU inference** | **~1 s** |
| **Total** | **~53 s** |

Only about **one second of every ~53 was GPU**. Average GPU utilization was ~7% (p50 = 0%). What looked like a
"GPU workload" was actually **CPU/IO-bound**: a million boxes per evaluation churned through Python NMS,
per-image postprocessing, and a ~170 MB JSON file that was written and then immediately re-read.

The second hotspot was **calibration**. Every quantizer round-tripped its running min/max to the host on every
batch; with on the order of 1,500 quantizers that's a host↔device **sync storm** — tens of thousands of syncs in
a single calibration pass, each one draining the GPU pipeline and serializing the batches behind it.

## What "faster" was allowed to mean

This is the part that's easy to skip and wrong to skip. "Faster" splits into two incompatible meanings:

- **Same answer, faster** — bit-identical output to the slow run. Provable. Safe to ship.
- **Good-enough answer, faster** — a smarter search that skips work. Bigger win, but *I* own the accuracy risk.

I asked which one the customer actually needed. The answer split the work into two deliverables — and it's why
the story ends with a *win I shipped* and a *win I didn't*.

## The fixes (shipped) — all bit-exact

Every change below preserves the exact final configuration and AP. The discipline was **AP-bit-exactness**: a
speedup that changes the answer isn't a speedup, it's a regression.

| Change | Phase | Effect |
|---|---|---|
| **2-GPU sharded calibration + parallel evals** — one validator per GPU; calibration shards merged (min/max are exact, order-independent reductions, so the merge is bit-identical); independent evaluations run in parallel | calibration + every eval | ~2× device overlap |
| **Compiled NMS** — identical float32 IoU arithmetic, identical tie-breaking order | eval postprocess | 16.4 s → 2.4 s |
| **Batched postprocess** — two bulk device→host copies instead of a small copy per image | eval postprocess | ~2.5× |
| **Drop the 170 MB JSON + cache ground truth** — pass predictions in-memory to the evaluator; cache the GT object instead of re-loading per call | eval | ~11 s removed |
| **On-device running min/max** — accumulate calibration stats on-device, finalize exactly once | calibration | removes the sync storm |
| **Cached ReLU6 bound + dead-buffer removal** — eliminate ~100 host↔device syncs per forward | every forward | ~100 syncs/forward removed |
| **Fused quantize/dequantize kernel (opt-in)** — one elementwise kernel instead of ~7 per q/dq pair; bit-exact by construction (power-of-two scale, single +0.5 rounding) | every forward | 16 m 09 s → 14 m 12 s |

Because every change is AP-invariant, the speedup carries no accuracy cost — and, checked directly, the
keep/revert decisions and every ladder verdict are unchanged. The implementation of these — the fused Q/DQ kernel
and the rest of the bit-exact compiler-side co-design changes, with per-forward budgets and invariants — is the
subject of [Fusing the Q/DQ chain, and other bit-exact compiler-side co-design wins](qdq-fusion-and-codesign.md).

## The numbers

Same task, same decision metric (AP@0.5, acceptance floor 0.69), same final configuration (all 60 feature-map
layers + 37/49 weight layers at 8-bit):

| Run | Wall | Per-iteration (median) |
|---|---|---|
| Baseline — 1 GPU, serial (Jan) | 2 h 15 m | 117 s |
| Serial, current code (control) | 52 m | ~46 s |
| **Parallel, 2-GPU (shipped)** | **16 m 09 s** | **8 s** |
| **Parallel + fused q/dq (shipped)** | **14 m 12 s** | 8 s |

- **3.18×** — parallel vs serial on the *same* codebase, bit-identical result. The clean, apples-to-apples number.
- **8.4×** — vs the January baseline (**9.5×** with the fused kernel).
- **14.6×** — per sensitivity iteration, 117 s → 8 s.
- **Bit-identical** — same final configuration, same AP, same ladder verdicts. A diff you can run, not a promise.

## The one I didn't ship

I also built a **proxy-guided search**: measure per-layer quantization distortion in a single reference forward
(zero full evaluations), fit a cost model, solve the bit-width selection exactly off-GPU, and confirm with one
or two full evaluations. It ran **14 full evaluations instead of 67** — ~5.4× faster than the serial control —
and it explored a *larger* space (weights and activations jointly, rather than one at a time).

On paper: ship it.

I didn't. Measured against the metric the customer actually lives by — **AP, not wall time** — it landed just
under the acceptance floor (about 0.004 below). The cause was instructive: accuracy damage on this detector is
**super-additive**. A few critical layers — and their interactions — dominate, so a per-layer model that prices
layers independently under-predicts any subset that contains a critical layer; and the baseline margin was thin
to begin with. (One of the probe configurations I measured actually *cleared* the bar; the strategy only returns
its confirmed solution, so it didn't.)

So I kept the exhaustive search as the accuracy-critical default and made the fast search **opt-in**, documented
exactly where it holds and where it doesn't. The decision to *withhold* a 5.4× number that would have quietly
missed the customer's bar is the part I'd be proudest of in a review.

## What "make it faster" actually taught me

The brief was three words. The work was four things:

1. **Diagnose before optimizing.** The slowness wasn't in any one function — it was that the "GPU" task was
   CPU/IO-bound, and that calibration had a sync storm.
2. **Scope the ambiguity.** "Faster" split into *same answer* vs *good enough*, and the customer's answer
   determined everything.
3. **Ship the provable win.** 3.18× (8.4–9.5×), bit-identical, diff-able.
4. **Hold the line on the risky one.** Measure against the metric the customer lives by, and say no to a faster
   number that misses the bar.

If you take one line: when a customer says "make it faster," the deliverable isn't a faster number. It's a faster
number you can **defend** — and an honest map of where the faster number stops being safe.

More and more of my work has taken this shape: a vague customer ask, real production constraints, a provable win,
and a judgment call on what *not* to ship. I've been calling it performance work. It reads, to me, like field work.

---

## How the numbers were measured

- **Apples-to-apples.** Parallel vs serial ran on the same codebase and the same inputs; the January baseline is
  the customer's prior reference run.
- **Bit-identical, end-to-end.** The final configuration, the AP, and every keep/revert and ladder verdict were
  diffed and match across runs — the speedup changes only the wall time to reach the *same* answer.
- **Wall time** is first-to-last timestamp inside the tuning log (the A/B control figures are orchestrator wall,
  same basis).
- Model and codebase identifiers are withheld; the workload is a small-input SSD/FPN-style object detector with
  ~1M boxes per evaluation.
