# Triton Flash Attention Tutorial Deep Dive

**Date:** 2026-03-26
**Source file:** `triton-lang/triton` repo, `python/tutorials/06-fused-attention.py`

---

## 1. Source Code Location and Overview

The canonical Triton Flash Attention tutorial lives at:
- **GitHub:** https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
- **Documentation:** https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

It implements **Flash Attention v2** (Tri Dao) in Triton with both forward and backward passes, supporting FP16 and FP8 (float8_e5m2), with autotune configs for Hopper, Blackwell, and AMD HIP targets.

---

## 2. Complete Forward Pass Inner Loop (`_attn_fwd_inner`)

This is the heart of the kernel -- the tile loop that iterates over K/V blocks and maintains online softmax state.

```python
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):

    # --- Determine iteration range based on STAGE ---
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M                     # off-diagonal (before current Q block)
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M  # diagonal block (causal mask needed)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3 (non-causal)
        lo, hi = 0, N_CTX                                 # full sequence

    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    # === TILE LOOP: iterate over K/V blocks ===
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Step 1: Load K block and compute QK^T --
        k = desc_k.load([offsetk_y, 0]).T        # [HEAD_DIM, BLOCK_N] -> transposed
        qk = tl.dot(q, k)                        # [BLOCK_M, BLOCK_N] = Q @ K^T

        # -- Step 2: Apply causal mask + online softmax max update --
        if STAGE == 2:  # diagonal block: needs causal mask
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:           # off-diagonal or non-causal: no mask needed
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # -- Step 3: Compute softmax numerator --
        p = tl.math.exp2(qk)         # exp2 instead of exp (uses qk_scale * log2(e))

        # -- Step 4: Compute correction factor for rescaling --
        alpha = tl.math.exp2(m_i - m_ij)   # correction: how much to scale old accumulators
        l_ij = tl.sum(p, 1)                # partial softmax denominator for this block

        # -- Step 5: Rescale output accumulator --
        # (special path for Blackwell warp specialization)
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]        # standard rescaling

        # -- Step 6: Load V block --
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T   # FP8: transposed layout
        else:
            v = desc_v.load([offsetv_y, 0])     # FP16: standard layout

        # -- Step 7: V matmul -- accumulate P @ V into output --
        p = p.to(dtype)                         # cast to working precision before dot
        acc = tl.dot(p, v, acc)                 # acc += P @ V (fused multiply-add)

        # -- Step 8: Update running statistics --
        # (placed at end of loop to reduce register pressure)
        l_i = l_i * alpha + l_ij    # update softmax denominator
        m_i = m_ij                   # update running maximum

        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    return acc, l_i, m_i
```

### Key Observations

1. **exp2 instead of exp**: The kernel uses `tl.math.exp2()` throughout. The `qk_scale` is pre-multiplied by `1.44269504` (= log2(e) = 1/ln(2)) so that `exp2(x * scale * log2e)` = `exp(x * scale)`. This is a hardware optimization -- `exp2` maps to a single PTX instruction on NVIDIA GPUs.

2. **Correction factor `alpha`**: `alpha = exp2(m_i - m_ij)` is always <= 1.0 (since m_ij >= m_i by definition of max). It rescales all previous accumulated work to account for the new global maximum.

3. **`acc = tl.dot(p, v, acc)`**: The third argument to `tl.dot` is the accumulator. This is a fused multiply-add: `acc = p @ v + acc`. It maps directly to Tensor Core MMA instructions.

4. **Register pressure management**: `l_i` and `m_i` updates are placed at the end of the loop body to minimize live register count during the compute-heavy dot products.

---

## 3. Main Forward Kernel (`_attn_fwd`)

```python
@triton.autotune(configs=list(filter(keep, configs)),
                 key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              FP8_OUTPUT: tl.constexpr, STAGE: tl.constexpr,
              warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):

    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)      # which Q block this CTA handles
    off_hz = tl.program_id(1)       # batch*head index
    off_z = off_hz // H
    off_h = off_hz % H

    # --- Create or reuse tensor descriptors ---
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    # V layout differs for FP8 (transposed)
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim],
                                         strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM],
                                     strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # === FP32 ACCUMULATOR INITIALIZATION ===
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running max per row
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # running softmax denom
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)        # output accumulator

    # === SCALE CONVERSION: multiply by log2(e) for exp2 optimization ===
    qk_scale = sm_scale
    qk_scale *= 1.44269504   # = log2(e) = 1/ln(2)

    # Load Q block once -- stays in SRAM for entire computation
    q = desc_q.load([qo_offset_y, 0])

    # === TWO-STAGE PROCESSING FOR CAUSAL ATTENTION ===
    # STAGE=3 (causal):  calls inner with STAGE=1 (off-diagonal), then STAGE=2 (diagonal)
    # STAGE=1 (non-causal): calls inner with STAGE=3 (full range)

    if STAGE & 1:  # off-band or non-causal
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        warp_specialize, IS_HOPPER)

    if STAGE & 2:  # on-band (diagonal with causal mask)
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        warp_specialize, IS_HOPPER)

    # === EPILOGUE: finalize softmax and store ===
    m_i += tl.math.log2(l_i)        # log-sum-exp for backward pass
    acc = acc / l_i[:, None]          # normalize by softmax denominator

    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)            # store logsumexp for backward
    desc_o.store([qo_offset_y, 0], acc.to(dtype))  # store output
```

---

## 4. Online Softmax Algorithm -- Mathematical Derivation

### Problem Statement

Standard softmax requires two passes over the data:
1. Find max and compute denominator: `m = max(x)`, `l = sum(exp(x - m))`
2. Compute output: `softmax(x_i) = exp(x_i - m) / l`

Flash Attention needs to compute `O = softmax(QK^T) @ V` without materializing the N x N attention matrix.

### Online Softmax Recurrence

When processing blocks sequentially, maintain running statistics `(m_i, l_i, O_i)`:

**Initialization:**
```
m_0 = -inf
l_0 = 0  (or 1.0 in the Triton impl for numerical reasons)
O_0 = 0  (zero matrix)
```

**Per-block update** (processing K_j, V_j block):

```
# 1. Compute local scores
S_ij = Q_i @ K_j^T                    # [BLOCK_M, BLOCK_N]

# 2. Local block maximum
m_ij = max(S_ij, axis=1)              # [BLOCK_M]

# 3. New global maximum
m_new = max(m_old, m_ij)              # [BLOCK_M]

# 4. Correction factor for previous accumulations
alpha = exp(m_old - m_new)            # <= 1.0 always

# 5. Local softmax numerators
P_ij = exp(S_ij - m_new[:, None])     # [BLOCK_M, BLOCK_N]

# 6. Local softmax partial denominator
l_ij = sum(P_ij, axis=1)             # [BLOCK_M]

# 7. Update global denominator with correction
l_new = l_old * alpha + l_ij

# 8. Rescale output accumulator and add new contribution
O_new = O_old * alpha[:, None] + P_ij @ V_j

# 9. Advance state
m_old = m_new
l_old = l_new
O_old = O_new
```

**Finalization** (after all blocks):
```
O_final = O / l[:, None]             # normalize by total denominator
logsumexp = m + log(l)               # for backward pass
```

### Why the Rescaling is Correct

The key identity: for concatenated vectors `x = [x1, x2]`:
```
m(x) = max(m(x1), m(x2))
l(x) = exp(m(x1) - m(x)) * l(x1) + exp(m(x2) - m(x)) * l(x2)
```

The correction factor `exp(m_old - m_new)` adjusts all previously accumulated exponentials from the old reference maximum to the new one:
```
exp(x_i - m_old) * exp(m_old - m_new) = exp(x_i - m_new)
```

This is mathematically exact -- no approximation involved.

### The exp2/log2(e) Optimization

Instead of `exp(x)`, the kernel computes `exp2(x * log2(e))`:
```
exp(x) = 2^(x * log2(e))
```

The scale factor `sm_scale` is pre-multiplied by `1.44269504` (= log2(e)):
```python
qk_scale = sm_scale * 1.44269504
```

Then all exponentials use `tl.math.exp2()` which maps to a single `ex2.approx` PTX instruction, faster than the multi-instruction `exp()`.

---

## 5. FP32 Accumulator Pattern for Numerical Stability

### The Pattern

All three running statistics are maintained in fp32 regardless of input dtype:

```python
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # max tracker
l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0            # denominator
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)        # output
```

### Why fp32 is Critical

1. **Accumulation error**: The output accumulator sums many small contributions across all K/V blocks. In fp16, each addition loses precision. Over N_CTX/BLOCK_N iterations, errors compound. fp32 gives 23 bits of mantissa vs. fp16's 10 bits.

2. **Softmax stability**: The correction factor `alpha = exp2(m_i - m_ij)` can be very small (close to zero) when the maximum increases significantly. fp16 would flush these to zero, losing all previous computation.

3. **The conversion point**: Attention weights `p` are cast down to working dtype (`p = p.to(dtype)`) only immediately before the V matmul, so the Tensor Core operates at the input's native precision, but the accumulation target remains fp32.

4. **Final output**: Only at the very end is the accumulator converted: `acc.to(dtype)`.

### In the Dao-AILab Implementation

The same pattern appears with explicit naming:
```python
acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
```

And uses `out_dtype=tl.float32` in dot products.

---

## 6. Tensor Descriptors and Memory Layout

### Host Tensor Descriptors (Hopper/Blackwell)

On SM >= 9.0, the kernel uses hardware tensor descriptors (TMA -- Tensor Memory Accelerator):

```python
desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
```

**Block shapes** define the tile sizes for hardware-accelerated loads:
- Q: `[BLOCK_M, HEAD_DIM]` -- one Q tile per CTA
- K: `[BLOCK_N, HEAD_DIM]` -- iterated in the inner loop
- V: `[BLOCK_N, HEAD_DIM]` (FP16) or `[HEAD_DIM, BLOCK_N]` (FP8, transposed)
- O: `[BLOCK_M, HEAD_DIM]` -- one output tile per CTA

**desc.load([offset_y, 0])** triggers a TMA copy from global memory to shared memory with hardware-managed addressing.

### Fallback Path (Pre-Hopper)

When TMA is unavailable, raw pointers are passed and the kernel creates descriptors on-device:
```python
desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM],
                                 strides=[HEAD_DIM, 1],
                                 block_shape=[BLOCK_M, HEAD_DIM])
```

### FP8 V Transposition

For FP8, V is stored in transposed layout `[HEAD_DIM, N_CTX]` because Hopper cannot perform FP8 dot with a non-transposed second operand. Blackwell lifts this restriction.

### Memory Access Pattern

The kernel is **Q-stationary**: each CTA loads one Q block into registers/SRAM and streams K/V blocks through it. This maximizes reuse of Q (loaded once, used N_CTX/BLOCK_N times).

---

## 7. Tile Loop Pipeline and V Matmul

### Pipeline Stages

The autotune configs test `num_stages` in {2, 3, 4}:
```python
for s in NUM_STAGES_OPTIONS   # [2, 3, 4]
```

`num_stages` controls the software pipelining depth: how many K/V loads are in flight simultaneously. With `num_stages=3`, while the current iteration computes QK^T and P@V, the next two K/V tiles are being prefetched.

### The V Matmul

```python
p = p.to(dtype)           # cast attention weights to working precision
acc = tl.dot(p, v, acc)   # fused: acc = P @ V + acc
```

This maps to Tensor Core MMA (Matrix Multiply-Accumulate) instructions:
- Input A: `p` of shape [BLOCK_M, BLOCK_N] in fp16/fp8
- Input B: `v` of shape [BLOCK_N, HEAD_DIM] in fp16/fp8
- Accumulator C: `acc` of shape [BLOCK_M, HEAD_DIM] in **fp32**

The Tensor Core natively accumulates in fp32 even with fp16 inputs, so no precision is lost.

### Warp Specialization (Blackwell)

On Blackwell, `warp_specialize=True` in the `tl.range()` call enables a producer-consumer pattern:
- **Producer warps**: prefetch K/V blocks from global to shared memory
- **Consumer warps**: compute QK^T and P@V from shared memory

The special accumulator reshaping code handles the different register layout under warp specialization:
```python
if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]
    acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
    acc0 = acc0 * alpha[:, None]
    acc1 = acc1 * alpha[:, None]
    acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
```

---

## 8. Causal Masking Strategy

The kernel uses a **two-stage strategy** to handle causal masking efficiently:

### Stage Configuration
- **Non-causal** (STAGE=1 at call site): inner function gets STAGE=3, processes full range [0, N_CTX)
- **Causal** (STAGE=3 at call site): two passes:
  - STAGE=1 inner: processes [0, start_m * BLOCK_M) -- all positions strictly before the diagonal, no masking needed
  - STAGE=2 inner: processes [start_m * BLOCK_M, (start_m+1) * BLOCK_M) -- the diagonal block, applies causal mask

### Why Two Stages?
Separating off-diagonal from diagonal avoids a conditional check inside the hot loop. For most iterations (off-diagonal), no mask is needed. Only the single diagonal block pays the masking cost:

```python
if STAGE == 2:
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # -1e6 masks future positions
```

---

## 9. Log-Sum-Exp for Backward Pass

The epilogue computes the log-sum-exp (LSE) value needed by the backward pass:

```python
m_i += tl.math.log2(l_i)      # LSE = m + log2(l)
tl.store(m_ptrs, m_i)         # store to M tensor
```

In the backward pass, this stored LSE replaces the need to recompute softmax from scratch. The backward kernel uses it directly:
```python
m = tl.load(M + offs_m)       # load stored LSE
pT = tl.math.exp2(qkT - m[None, :])  # reconstruct attention weights
```

The LSE is stored in log-base-2 (consistent with the exp2 convention throughout).

---

## 10. Backward Pass Architecture

The backward pass uses three sub-kernels:

### `_attn_bwd_preprocess`
Computes delta values: `delta_i = sum(O_i * dO_i)` (element-wise row sums).

### `_attn_bwd_dkdv`
K-stationary loop: loads one K/V block, streams Q/dO blocks to compute dK, dV.
```python
# For each Q block:
qkT = tl.dot(k, qT)               # [BLOCK_N, BLOCK_M]
pT = tl.math.exp2(qkT - m[None, :])  # reconstruct attention weights
dv += tl.dot(ppT, do)             # dV += P^T @ dO
dpT = tl.dot(v, tl.trans(do))     # dP = V @ dO^T
dsT = pT * (dpT - Di[None, :])    # dS = P * (dP - delta)
dk += tl.dot(dsT, tl.trans(qT))   # dK += dS^T @ Q^T
```

### `_attn_bwd_dq`
Q-stationary loop: loads one Q/dO block, streams K/V blocks to compute dQ.
```python
# Pre-scales K by sm_scale * RCP_LN2
qk = tl.dot(q, kT)
p = tl.math.exp2(qk - m)
dp = tl.dot(do, vT)
ds = p * (dp - Di[:, None])
dq += tl.dot(ds, tl.trans(kT))
# Final: dq *= LN2 (= ln(2)) to undo the log2(e) scaling
```

The backward pass uses fixed block sizes: `BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32` with `BLK_SLICE_FACTOR=2` for the diagonal processing.

---

## 11. Autotune Configuration Space

```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w,
                  pre_hook=_host_descriptor_pre_hook)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]       # software pipeline depth
    for w in [4, 8]           # warps per CTA
]
```

**Pruning rules:**
- BLOCK_M must be <= N_CTX
- For causal: BLOCK_M >= BLOCK_N (ensures diagonal blocks work correctly)
- Hopper SM90: skip configs where BLOCK_M * BLOCK_N < 128*128 with 8 warps

**Pre-hook** adjusts tensor descriptor block shapes to match the selected config:
```python
def _host_descriptor_pre_hook(nargs):
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    ...
```

---

## 12. Performance Benchmarks (from tutorial)

Configuration: batch=4, heads=32

| HEAD_DIM | Mode | Causal | Seq Len | TFLOPS (Triton FP16) |
|----------|------|--------|---------|---------------------|
| 64 | fwd | True | 1024 | 112 |
| 64 | fwd | True | 16384 | 166 |
| 128 | fwd | True | 1024 | 122 |
| 128 | fwd | True | 16384 | 177 |
| 64 | bwd | True | 1024 | ~60 |
| 64 | bwd | True | 16384 | ~101 |

Backward pass achieves approximately 60% of forward throughput.

---

## Sources

- [Triton Fused Attention Tutorial (GitHub)](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)
- [Triton Fused Attention Tutorial (Documentation)](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton Flash Attention Kernel Walkthrough (Nathan Chen)](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html)
- [Understanding Flash Attention in Triton (Alex Dremov)](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [The Anatomy of a Triton Attention Kernel (arXiv 2511.11581)](https://arxiv.org/html/2511.11581v1)
- [Online Softmax Demystified (isztld.com)](https://isztld.com/posts/online-softmax.html)
- [From Online Softmax to FlashAttention (UW CSE599m)](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [The Basic Idea Behind Flash Attention (Damek Davis)](https://damek.github.io/random/basic-idea-behind-flash-attention/)
- [Dao-AILab Flash Attention Triton Implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
- [vLLM Triton Flash Attention](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_flash_attention.py)
- [Flash Attention v2 Paper (Tri Dao)](https://tridao.me/publications/flash2/flash2.pdf)
- [Online Softmax to Flash Attention (Matthew Gunton)](https://medium.com/data-science-collective/online-softmax-to-flash-attention-and-why-it-matters-9d676e7c50a8)
