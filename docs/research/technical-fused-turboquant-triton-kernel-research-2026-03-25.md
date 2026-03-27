---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'fused-turboquant-triton-kernel'
research_goals: 'Evaluate feasibility and path for a fused TQ4 dequant-attention Triton kernel on RTX 4090 targeting Molmo2 GQA'
user_name: 'Alberto-Codes'
date: '2026-03-25'
web_research_enabled: true
source_verification: true
---

# Fused TurboQuant Triton Kernel for Molmo2 on RTX 4090

**Date:** 2026-03-25
**Author:** Alberto-Codes
**Research Type:** Technical Feasibility + Implementation Plan

---

## Executive Summary

A fused TurboQuant dequant-attention Triton kernel for Molmo2 on RTX 4090 is **feasible and achievable in 2-3 days** by vendoring and adapting the Dejan.ai Triton kernel, which already handles GQA and was tested on RTX 4090 with Gemma 3.

**Current state:** Our `CompressedDynamicCache` achieves 3.76x KV cache compression (TQ4 nibble-packed) with near-identical output quality, but is **3.36x slower** because it dequantizes the entire cache at every layer at every generation step.

**Target:** A fused kernel that computes `Q @ compressed_K^T` directly from packed indices would reduce per-layer memory traffic from ~25 MB to ~3 MB, projecting a **2.8-3.9x decode speedup** (from 8.9 tok/s to 25-35 tok/s) — eliminating the overhead entirely.

**Recommended path:** Vendor Dejan.ai's fused Triton kernel, adapt for Molmo2's 7:1 GQA (28 query / 4 KV heads), add nibble unpacking for our TQ4 packed indices, and register via HuggingFace's `AttentionInterface.register()` API.

**Key discoveries:**
- Molmo2-4B uses 28/4 GQA (7:1 ratio), not 32/8 — confirmed from model config
- The 7:1 GQA ratio **amplifies** compression benefit (7 query heads share each compressed KV read)
- HuggingFace has a first-class `AttentionInterface.register()` API — no monkey-patching needed
- `torch.compile` cannot auto-fuse this — a custom Triton kernel is required
- Dejan.ai's pre-rotation trick (`q_rot = Q @ Pi_T`) eliminates the 128x128 rotation from the kernel inner loop

---

## Table of Contents

1. [Technology Stack Analysis](#technology-stack-analysis) — Existing kernels, Molmo2 internals, RTX 4090 roofline, torch.compile feasibility
2. [Integration Patterns Analysis](#integration-patterns-analysis) — HuggingFace attention registration, Dejan.ai integration pattern, vLLM backend
3. [Architectural Patterns and Design](#architectural-patterns-and-design) — Fused kernel architecture, block structure, Molmo2 adaptations, risk assessment
4. [Implementation Plan](#implementation-plan) — 3-phase plan with day-by-day tasks, success criteria, expected performance

---

## Research Overview

Technical research into implementing a fused TurboQuant dequantization-attention Triton kernel for RTX 4090 (Ada Lovelace SM89) targeting Molmo2 vision-language models with Grouped Query Attention.

---

## Technical Research Scope Confirmation

**Research Topic:** Fused TurboQuant Triton Kernel for RTX 4090 + Molmo2 GQA
**Research Goals:** Evaluate feasibility and path for a fused TQ4 dequant-attention kernel on RTX 4090 targeting Molmo2 GQA

**Technical Research Scope:**

- Existing Kernel Analysis — Dejan.ai, tonbistudio, Kitty, BitDecoding Triton kernels
- Molmo2 Attention Internals — GQA, RoPE, hook points in modeling_molmo2.py
- RTX 4090 Roofline Analysis — bandwidth ceiling, shared memory, speedup bounds
- torch.compile Feasibility — auto-fusion evaluation
- Vendor vs Build Decision Matrix — synthesis and recommendation

**Research Methodology:**

- Current web data with rigorous source verification
- Local codebase analysis (Molmo2 model code, turboquant-consumer)
- Multi-source validation for critical technical claims
- Confidence level framework for uncertain information
- Comprehensive technical coverage with architecture-specific insights

**Scope Confirmed:** 2026-03-25

---

## Technology Stack Analysis

### 1. Existing Triton/CUDA Kernels for Quantized KV Cache Attention

#### Dejan.ai TurboQuant Triton Kernel (PRIMARY CANDIDATE)

**Source:** [dejan.ai/blog/turboquant](https://dejan.ai/blog/turboquant/) — downloadable at `dejan.ai/media/code/turboquant.zip`

| Property | Detail |
|---|---|
| GQA support | **Yes** — maps `kv_head = q_head // gqa_ratio` explicitly in kernel |
| Bit widths | 2-bit, 4-bit tested |
| Packing format | uint8 indices + fp16 norms; **no nibble packing** (noted as future work) |
| Fused dequant+attention | **Yes** — computes Q@K^T directly from compressed uint8 key indices via centroid table lookups, never materializes fp16 keys |
| GPU tested | RTX 4090 (Gemma 3 4B, 8 Q heads / 4 KV heads) |

**Critical design insight:** The kernel exploits orthogonality of the rotation matrix to avoid rotating keys inside the kernel. Since `<q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>`, the **query is pre-rotated once** outside the kernel (`q_rot = q @ Q_T`), and the kernel operates on pre-rotated queries against un-rotated centroid values. The rotation matrix (128x128x4 = 64 KB) is never loaded in the kernel's inner loop.

**Kernel structure:** Computes Q@K^T only (not softmax or softmax@V). Materializes the full attention score matrix `[BH_q, kv_len]`. Not Flash Attention-style tiling.

**Adaptation cost for Molmo2:**

| Change | Effort |
|--------|--------|
| GQA ratio (7:1 for Molmo2 vs 2:1 for Gemma3) | Trivial — change n_q_heads/n_kv_heads params |
| Nibble unpacking in kernel inner loop | ~20 lines of Triton (shift + mask) |
| fp32 norms (vs fp16 in Dejan) | Trivial dtype change |
| RoPE integration (swap Gemma3 RoPE for Molmo2) | Medium — different apply_rotary_pos_emb |
| Module patching (Gemma3 → Molmo2 attention class) | Medium — different class/method names |
| **Total drop-in (no nibble pack)** | **1-2 days** |
| **With nibble unpacking** | **+1 day** |
| **Full Flash Attention fusion** | **+1-2 weeks** (essentially new kernel) |

_Source: [Dejan.ai blog](https://dejan.ai/blog/turboquant/), code archive analysis_

#### Other Kernel Candidates

| Kernel | GQA | Quant KV | Fused | Language | Adaptation Cost |
|--------|:---:|:---:|:---:|---------|------|
| **KIVI** | Yes | INT2/INT4 | Hybrid (Triton quant + CUDA fused matmul) | Triton + CUDA | Medium — fused matmul is CUDA, not Triton |
| **BitDecoding** | Yes | INT2/INT4 | Fully fused (CUDA/PTX) | Pure CUDA | High — deep PTX ops (ldmatrix, lop3, wgmma), not portable to Triton |
| **Kitty** | Yes | INT2/INT4 mixed | Separate dequant + attention | Triton (dequant), standard (attention) | Medium — different quant scheme (dense-sparse, not VQ) |
| **tonbistudio/turboquant-pytorch** | No | N/A | No kernel | Pure PyTorch | N/A — reference implementation only, no Triton |
| **FlashInfer** | Yes | FP4/FP8 | Fused (CUDA) | CUDA | High — different quant format, not VQ |

_Sources: [KIVI](https://github.com/jy-yuan/KIVI), [BitDecoding](https://github.com/DD-DuDa/BitDecoding), [Kitty](https://github.com/Summer-Summer/Kitty), [FlashInfer](https://github.com/flashinfer-ai/flashinfer)_

**Verdict:** Dejan.ai is the clear starting point — only fully-Triton fused TurboQuant attention kernel with GQA, tested on RTX 4090.

---

### 2. Molmo2 Attention Internals

**Source:** `modeling_molmo2.py` from HuggingFace cache

> **CORRECTION:** Molmo2-4B uses **28 query heads / 4 KV heads** (7:1 GQA ratio), not 32/8 as previously assumed. Confirmed from `configuration_molmo2.py`.

| Parameter | Molmo2-4B | Molmo2-8B |
|-----------|-----------|-----------|
| Query heads | 28 | 32 (TBC) |
| KV heads | 4 | 8 |
| GQA ratio | 7:1 | 4:1 |
| head_dim | 128 | 128 |
| Layers | 36 | 36 |

**Attention implementation:** Pluggable architecture via `config._attn_implementation`:
- **Default (`eager`):** Manual `torch.matmul(query, key.T) * scaling` → softmax → `torch.matmul(attn_weights, value)`. Does NOT use `F.scaled_dot_product_attention`.
- **SDPA mode:** Uses `F.scaled_dot_product_attention` (dispatches to Flash Attention 2 on GPU).
- **Flash Attention 2 mode:** Direct FA2 call.

**GQA broadcast:** Uses `repeat_kv()` to expand KV heads from (B, 4, T, 128) → (B, 28, T, 128) before matmul. A fused kernel could avoid this expansion entirely.

**RoPE application order:** RoPE is applied **BEFORE** cache update. The cache stores already-rotated keys. This means the fused kernel receives pre-rotated keys — the Q pre-rotation trick from Dejan.ai's kernel still applies, but the cached keys are already in the rotated domain.

Wait — this creates a conflict. Dejan.ai's kernel assumes keys are **unrotated** (raw centroid values) and the query is pre-rotated to compensate. But Molmo2 applies RoPE to keys **before** caching, so our compressed indices represent **RoPE-rotated** key vectors, not raw projections.

**Resolution:** This actually works fine. The TurboQuant rotation (Haar-random orthogonal Π) is separate from RoPE. The flow is:
1. Model computes K projection
2. RoPE is applied: `K_rope = apply_rope(K)`
3. Cache stores K_rope
4. Our compressor quantizes K_rope: `indices, norms = quantize(K_rope)` — the TurboQuant rotation Π operates on whatever vector it receives, RoPE or not.
5. At dequant time: `K_hat = dequantize(indices, norms)` ≈ K_rope
6. The fused kernel computes `Q @ K_hat^T` where both Q and K_hat are in the RoPE-rotated domain

The Dejan.ai pre-rotation trick (`Q_rot = Q @ Pi_T`) then applies within the TurboQuant domain, not the RoPE domain. The two rotations are independent and composable.

**Best hook point:** Replace `eager_attention_forward()` function (lines 593-616 in modeling_molmo2.py). This is called from `Molmo2Attention.forward()` via the pluggable interface. A new `_attn_implementation = "fused_compressed"` variant could be added cleanly.

_Source: Local analysis of `modeling_molmo2.py` from HuggingFace cache_

---

### 3. RTX 4090 Roofline Analysis

| Metric | RTX 4090 (Ada SM89) | H100 SXM (Hopper SM90) | Ratio |
|--------|---------------------|------------------------|-------|
| Memory bandwidth | 1,008 GB/s (GDDR6X) | 3,350 GB/s (HBM3) | 3.3x |
| FP16 TFLOPS | ~330 | ~990 | 3.0x |
| L2 cache | 72 MB | 50 MB | 1.44x (4090 wins) |
| Shared memory/SM | 128 KB | 228 KB | 0.56x |
| SMs | 128 | 132 | ~1x |

**Achievable Triton bandwidth on RTX 4090:**

| Kernel type | Achieved bandwidth | % of theoretical |
|------------|-------------------|-----------------|
| Simple streaming (elementwise) | 850-950 GB/s | 84-94% |
| Reductions (layernorm, softmax) | 700-850 GB/s | 70-84% |
| Attention-like (mixed compute + memory) | 500-750 GB/s | 50-74% |

_Confidence: MEDIUM — based on community benchmarks and Triton documentation through early 2025_

**Triton Flash Attention on RTX 4090:**
- Triton FA: 120-170 TFLOPS (65-85% of CUDA FA2)
- CUDA FA2: 180-220 TFLOPS
- Gap narrows for longer sequences (>4096)

_Confidence: MEDIUM — from benchmark publications through early 2025_

**Roofline for fused TQ4 attention (decode phase, batch=1):**

During decode, the kernel reads compressed keys for ALL cached tokens to compute attention for ONE new query token. This is heavily **memory-bound**.

**Current unfused path (our 3.36x overhead):**
1. Read compressed indices: 1 × 4 KV heads × 11K tokens × 64 bytes (nibble-packed) = 2.8 MB
2. Read fp32 norms: 1 × 4 × 11K × 4 bytes = 176 KB
3. Dequantize (rotation matmul): 11K × 4 × 128² FLOPs = 72M FLOPs per layer
4. Write decompressed K: 1 × 4 × 11K × 128 × 2 bytes = 11.2 MB
5. Read Q: 1 × 28 × 1 × 128 × 2 bytes = 7 KB
6. Read decompressed K (again for matmul): 11.2 MB (expanded to 28 heads via repeat_kv)
7. Compute Q@K^T: negligible FLOPs at batch=1

Total memory traffic: ~2.8 MB (read compressed) + ~11.2 MB (write decompressed) + ~11.2 MB (read for matmul) = **~25 MB per layer**

**Fused kernel path (target):**
1. Read compressed indices: 2.8 MB
2. Read fp32 norms: 176 KB
3. Read Q (pre-rotated): 7 KB
4. Read centroids: 16 × 4 bytes = 64 bytes (fits in registers)
5. Compute: centroid lookup + dot product (no matmul, no full K materialization)

Total memory traffic: **~3 MB per layer** (8x less than unfused)

**Theoretical speedup ceiling on RTX 4090:**

At 800 GB/s effective bandwidth:
- Unfused: 25 MB / 800 GB/s = 31 µs per layer × 36 layers = **1.1 ms per decode step**
- Fused: 3 MB / 800 GB/s = 3.75 µs per layer × 36 layers = **0.135 ms per decode step**
- **Theoretical speedup: ~8x** (matching Google's H100 claim, because the speedup comes from reduced memory traffic, not raw bandwidth)

However, the fused kernel has compute overhead (centroid lookup per element) that the unfused path amortizes via batched matmul. Realistically: **4-6x speedup** on RTX 4090.

_Confidence: MEDIUM — roofline analysis from first principles, not measured_

**Optimal Triton block sizes for Ada Lovelace:**

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| BLOCK_M (query tokens) | 1 (decode) or 128 (prefill) | Batch=1 decode = 1 query token |
| BLOCK_N (KV tokens) | 64-128 | Tile over KV sequence |
| num_warps | 4 | Sweet spot for Ada attention |
| num_stages | 2 | Double-buffering; 3 may exceed shared memory |
| Shared memory target | < 64 KB per block | Allows 2 blocks/SM for occupancy |

_Source: Triton documentation, community benchmarks, Ada Lovelace architecture specs_

---

### 4. torch.compile Feasibility

**Can torch.compile auto-fuse our decompress + attention path?**

**No.** Evaluation:

| Scenario | Result |
|----------|--------|
| Custom Triton kernel via `torch.autograd.Function` | **Opaque to compiler** — treated as graph break, not fused with surrounding ops |
| Our CompressedDynamicCache.dequantize + standard attention | **Cannot fuse** — the dequant is inside cache.update(), which is a method call, not a traced operation |
| Standard model with `torch.compile(model)` | **Fuses non-attention ops** (layernorm, residual, MLP) but dispatches attention to SDPA → Flash Attention 2 |

**Verdict:** `torch.compile` cannot help here. The decompress-then-attend pattern requires a custom fused kernel. However, `torch.compile` WILL help fuse operations around the attention (layernorm, residual connections, MLP), providing independent speedups.

_Confidence: HIGH — well-documented limitation of torch.compile's Inductor backend_

---

### 5. Vendor vs Build Decision Matrix

| Path | Effort | Speedup | Risk | Recommendation |
|------|--------|---------|------|----------------|
| **A: Vendor Dejan.ai kernel + adapt for Molmo2** | 2-3 days | 2-4x over current | Low — proven kernel, just config changes + nibble unpack | **RECOMMENDED** |
| **B: Build fused Flash Attention kernel from scratch** | 1-2 weeks | 4-6x over current | Medium — requires Triton FA expertise | Phase 2 if A succeeds |
| **C: Wait for vLLM native TurboQuant** | 0 (wait) | Unknown | Low effort but unknown timeline | Passive monitoring |
| **D: Use torch.compile for auto-fusion** | 1 hour test | 0x (won't work) | N/A | **Eliminated** |

**Recommended path: A → B**

1. **Phase 1 (2-3 days):** Vendor Dejan.ai's `triton_attention.py` and `turboquant_fused.py`. Adapt for Molmo2: change GQA ratio (7:1), swap RoPE implementation, update module patching for Molmo2Attention class, add nibble unpacking to kernel inner loop.

2. **Phase 2 (1-2 weeks, optional):** If Phase 1 speedup is insufficient, build a Flash Attention-style fused kernel using the "Anatomy of a Triton Attention Kernel" (arXiv 2511.11581) as scaffold, injecting centroid lookup into the inner tile loop.

3. **Ongoing:** Monitor vLLM for native TurboQuant support.

---

### Research Synthesis

**Key finding:** The Dejan.ai TurboQuant Triton kernel is a directly vendorable starting point. It handles GQA, computes attention from compressed keys without decompression, and was tested on RTX 4090. The adaptation for Molmo2 (7:1 GQA, nibble-packed 4-bit, fp32 norms) is a 2-3 day project.

**Critical correction:** Molmo2-4B uses 28 query heads / 4 KV heads (7:1 ratio), not 32/8 as previously documented. Our benchmark data and compression stats are correct (they use actual model config), but the roadmap docs need updating.

**Theoretical speedup:** 4-6x on RTX 4090 (vs current 3.36x overhead), derived from 8x memory traffic reduction partially offset by per-element compute overhead.

**Sources:**
- [Dejan.ai TurboQuant Triton kernel](https://dejan.ai/blog/turboquant/)
- [KIVI](https://github.com/jy-yuan/KIVI) — GQA + INT2/4 fused dequant
- [BitDecoding](https://github.com/DD-DuDa/BitDecoding) — CUDA-only, 7.5x speedup
- [Kitty](https://github.com/Summer-Summer/Kitty) — mixed-precision 2-bit KV cache
- [Anatomy of Triton Attention Kernel](https://arxiv.org/abs/2511.11581)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — CUDA FP4/FP8 attention
- Molmo2 `modeling_molmo2.py` (local HuggingFace cache)
- RTX 4090 Ada Lovelace architecture specifications

---

## Integration Patterns Analysis

### HuggingFace Attention Registration (Recommended Path)

HuggingFace transformers provides a first-class API for custom attention implementations via `AttentionInterface.register()`:

```python
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

def turboquant_fused_attention(module, query, key, value, attention_mask, **kwargs):
    # Pre-rotate query, call Triton kernel, handle softmax + V matmul
    return attn_output, attn_weights

AttentionInterface.register("turboquant_fused", turboquant_fused_attention)
AttentionMaskInterface.register("turboquant_fused", sdpa_mask)

# Use at model load
model = AutoModelForImageTextToText.from_pretrained(
    "allenai/Molmo2-4B",
    attn_implementation={"text_config": "turboquant_fused"}  # Per-backbone!
)

# Or runtime switching (no reload)
model.set_attn_implementation("turboquant_fused")
```

**Critical:** Must register BOTH attention AND mask functions — omitting the mask causes silent incorrect computation (HuggingFace issue #40362).

**Multimodal support:** Molmo2 is an image-text model — `attn_implementation` accepts per-backbone dicts, so vision attention can stay as SDPA while text attention uses the fused kernel.

_Confidence: HIGH — official API, documented with examples_
_Source: [HuggingFace Attention Backends docs](https://huggingface.co/docs/transformers/main/attention_interface), [Issue #40362](https://github.com/huggingface/transformers/issues/40362)_

### PyTorch Triton Op Registration

`torch.library.triton_op` (PyTorch 2.6+) makes Triton kernels first-class PyTorch ops:

```python
from torch.library import triton_op, wrap_triton

@triton_op("turboquant::fused_dequant_attn", mutates_args={})
def fused_dequant_attn(q_rotated, key_indices, key_norms, centroids, scale):
    out = torch.empty(...)
    wrap_triton(fused_qk_kernel)[grid](q_rotated, key_indices, ...)
    return out
```

Benefits: `torch.compile` tracing, CPU fallback registration, autograd support, AOTInductor export.

_Source: [PyTorch triton_op tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)_

### Dejan.ai Integration Pattern (Proven on RTX 4090)

The Dejan.ai implementation patches at the **attention layer level** (not the cache level like our current approach):

| Step | Operation | Location |
|------|-----------|----------|
| 1 | Q/K/V projections | Standard linear layers |
| 2 | RoPE application | Pre-rotation on Q and K |
| 3 | Compress keys | `cache.store_compressed_key()` — quantize K to uint8 indices + norms |
| 4 | Store values | Standard fp16 in DynamicCache (values not compressed) |
| 5 | **Pre-rotate query** | `q_rot = query @ Q_T` — **moves rotation cost OUT of kernel** |
| 6 | **Fused kernel** | `fused_qk_scores(q_rot, indices, norms, centroids, scale)` |
| 7 | Softmax + V matmul | Standard PyTorch (not fused) |

**Key insight:** The kernel exploits `<q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>` to avoid the expensive 128x128 rotation inside the inner loop. The query is pre-rotated ONCE, and the kernel does simple centroid gather + dot product.

**Kernel signature:**

```python
def fused_qk_scores(
    q_rotated: torch.Tensor,    # [batch, n_q_heads, q_len, head_dim]
    key_indices: torch.Tensor,  # [batch, n_kv_heads, kv_len, head_dim] uint8
    key_norms: torch.Tensor,    # [batch, n_kv_heads, kv_len]
    centroids: torch.Tensor,    # [n_levels] float32
    scale: float,               # 1/sqrt(head_dim)
) -> torch.Tensor:              # [batch, n_q_heads, q_len, kv_len]
```

_Source: [Dejan.ai blog](https://dejan.ai/blog/turboquant/), code archive analysis_

### Our Current Integration (CompressedDynamicCache) vs Target

| Aspect | Current (cache-level patch) | Target (attention-level patch) |
|--------|---------------------------|-------------------------------|
| Patched target | `DynamicCache.update()` | `Molmo2Attention.forward()` |
| Compression | On cache write | On cache write (same) |
| Dequantization | Full tensor, every layer, every step | **Never** — kernel reads compressed directly |
| Overhead source | 128x128 rotation matmul per vector | Eliminated by pre-rotating Q once |
| Integration style | Non-invasive (cache wrapper) | `AttentionInterface.register()` |

### vLLM Backend Plugin (Future Path)

vLLM has a formal `register_backend()` API for custom attention backends:

```python
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum

@register_backend(AttentionBackendEnum.CUSTOM)
class TurboQuantAttentionBackend:
    ...
```

This would be the production deployment path once the fused kernel is validated via HuggingFace.

_Source: [vLLM Backend Registry](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py), [vLLM Plugin docs](https://docs.vllm.ai/en/latest/design/plugin_system/)_

### Recommended Integration Sequence

1. **Phase 1:** Vendor Dejan.ai kernel + register via `AttentionInterface.register("turboquant_fused", ...)` for Molmo2
2. **Phase 2:** Wrap kernel with `@triton_op("turboquant::fused_dequant_attn")` for torch.compile compatibility
3. **Phase 3:** Port to vLLM `register_backend()` for production serving

---

## Architectural Patterns and Design

### Fused Kernel Architecture

The fused TQ4 dequant-attention kernel replaces three separate operations (dequantize + GQA broadcast + Q@K^T matmul) with a single GPU kernel pass:

**Current architecture (unfused, 3.36x overhead):**
```
Per layer, per decode step:
  1. Read compressed indices (uint8 nibble-packed) + fp32 norms
  2. Unpack nibbles → int64 indices
  3. Centroid lookup → float32 values
  4. Rotation matmul: (N, 128) @ (128, 128) → decompressed keys
  5. Cast to target dtype
  6. GQA broadcast: (B, 4, T, 128) → (B, 28, T, 128)  [7x memory expansion]
  7. Q @ K^T matmul → attention scores

  Memory traffic: ~25 MB per layer (read compressed + write/read decompressed)
```

**Target architecture (fused):**
```
Per layer, per decode step:
  1. Pre-rotate query ONCE: q_rot = Q @ Pi_T    [outside kernel, one matmul]
  2. Kernel reads: compressed indices + norms + pre-rotated query + centroids
  3. Inner loop: unpack nibble → centroid lookup → dot with q_rot → accumulate
  4. Write: attention scores (B, 28, 1, T)

  Memory traffic: ~3 MB per layer (read compressed only, no decompressed write)
```

**Why this works:** The rotation `R^T @ centroids[idx]` becomes `centroids[idx]` (unrotated values) when queries are pre-rotated by `R`. The dot product `<R @ q, centroids[idx]>` is computed element-wise in the kernel without materializing the full decompressed key vector. GQA is handled by the kernel's thread mapping (`kv_head = q_head // gqa_ratio`) — no memory expansion needed.

### Kernel Block Structure (Ada Lovelace Optimized)

Based on RTX 4090 constraints (128 KB shared memory, 1008 GB/s bandwidth):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid | `(batch * n_q_heads, cdiv(kv_len, BLOCK_S))` | One program per query-head × KV-seq tile |
| BLOCK_S | 64-128 | KV sequence tile size; auto-tuned |
| BLOCK_D | 128 | Full head_dim in one tile (fits in registers) |
| num_warps | 4 | Sweet spot for Ada attention kernels |
| num_stages | 2 | Double-buffering; 3 may exceed shared memory |
| Shared memory | < 64 KB per block | Allows 2 concurrent blocks per SM |

**Inner loop pseudocode:**
```
for each BLOCK_S tile of KV sequence:
    load packed_indices[kv_head, tile, head_dim//2]  # uint8, nibble-packed
    load norms[kv_head, tile]                         # fp32

    # Unpack nibbles
    hi = packed >> 4        # even indices
    lo = packed & 0x0F      # odd indices

    # Centroid gather (16 centroids fit in registers)
    k_vals_even = centroids[hi]
    k_vals_odd  = centroids[lo]

    # Interleave back to head_dim order
    # Dot with pre-rotated query
    acc += sum(k_vals * q_rot)

    # Scale by norms
    scores[tile] = norms * acc * scale
```

### Molmo2-Specific Adaptations

| Aspect | Gemma 3 (Dejan.ai) | Molmo2-4B (target) | Change Required |
|--------|--------------------|--------------------|-----------------|
| GQA ratio | 2:1 (8Q/4KV) | 7:1 (28Q/4KV) | Config param change |
| head_dim | 256 | 128 | Config param change |
| RoPE | Gemma3 style | Molmo2 `apply_rotary_pos_emb` | Swap function in patched forward |
| Attention class | `Gemma3Attention` | `Molmo2Attention` | Adapt module patching |
| Vision backbone | N/A | Separate ViT attention (don't touch) | Apply only to text_config |
| QK LayerNorm | No | Optional (`use_qk_norm`) | Handle if enabled |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Nibble unpacking adds overhead that negates memory savings | Low | Medium | Profile before/after; bit ops are fast on GPU |
| 7:1 GQA ratio reduces benefit (each KV read serves 7 queries) | Low | Low | Actually AMPLIFIES benefit — 7 queries share same compressed KV read |
| Numerical precision drift from pre-rotated queries | Medium | High | Validate against unfused path with long-sequence regression test |
| Molmo2 QK LayerNorm interacts badly with pre-rotation | Low | Medium | Test with and without QK norm |
| Triton kernel performance on Ada << CUDA FA2 | Medium | Medium | Accept 65-85% of CUDA FA2 speed; still much faster than current unfused |

_Confidence: MEDIUM-HIGH — architecture is proven (Dejan.ai), risks are in adaptation, not fundamentals_

---

## Implementation Plan

### Phase 1: Vendor + Adapt Dejan.ai Kernel (2-3 days)

**Day 1: Vendor and validate baseline**

| Step | Task | Output |
|------|------|--------|
| 1.1 | Download Dejan.ai code archive from `dejan.ai/media/code/turboquant.zip` | 5 files in `vendored/dejan/` |
| 1.2 | Add `triton>=3.0` to turboquant-consumer dev dependencies | Updated `pyproject.toml` |
| 1.3 | Run Dejan.ai's `run_demo.py` on Gemma 3 4B (if available) or adapt for a small test model | Baseline correctness verified |
| 1.4 | Write unit test: `fused_qk_scores()` output matches `torch.matmul(Q, K.T) * scale` for random inputs | Test green |

**Day 2: Adapt for Molmo2**

| Step | Task | Output |
|------|------|--------|
| 2.1 | Copy `turboquant_fused.py` → create `molmo2_fused_attention.py` | New integration module |
| 2.2 | Swap GQA config: `n_q_heads=28, n_kv_heads=4` (7:1 ratio) | Config change |
| 2.3 | Replace Gemma3 RoPE with Molmo2's `apply_rotary_pos_emb()` | Function swap |
| 2.4 | Update module patching: target `Molmo2Attention` class, handle `use_qk_norm` | Module adaptation |
| 2.5 | Register via `AttentionInterface.register("turboquant_fused", ...)` | Clean integration |
| 2.6 | Add nibble unpacking to Triton kernel inner loop (20 lines): `hi = packed >> 4; lo = packed & 0xF` | Kernel modification |
| 2.7 | Change kernel norm loading from fp16 to fp32 | Dtype change |

**Day 3: Validate and benchmark**

| Step | Task | Output |
|------|------|--------|
| 3.1 | Run fused kernel on Molmo2-4B with Seinfeld clip (same setup as Experiment 004) | Experiment 005 |
| 3.2 | Compare output quality: fused vs unfused (cosine similarity, text comparison) | Quality validation |
| 3.3 | Measure speedup: tokens/sec, VRAM peak, time ratio | Performance validation |
| 3.4 | Run long-sequence regression test with fused path | Regression gate |
| 3.5 | Write experiment log | Documentation |

### Phase 2: PyTorch Op Registration (Optional, +1 day)

| Step | Task |
|------|------|
| 2.1 | Wrap kernel with `@triton_op("turboquant::fused_dequant_attn")` |
| 2.2 | Register CPU fallback (standard unfused path) |
| 2.3 | Test with `torch.compile(model)` to verify non-attention fusion still works |

### Phase 3: vLLM Backend (Future, blocked on vLLM API)

| Step | Task |
|------|------|
| 3.1 | Implement `TurboQuantAttentionBackend` extending vLLM's `AttentionBackend` |
| 3.2 | Register via `register_backend(AttentionBackendEnum.CUSTOM, ...)` |
| 3.3 | Validate with vLLM serving + Molmo2 + video workload |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Output quality | >0.999 cosine sim vs unfused | Must not degrade quality |
| Throughput (decode) | >1.0x baseline (faster, not slower) | Eliminate the 3.36x overhead |
| KV cache VRAM | Same as TQ4 unfused (3.76x compression) | Kernel doesn't change storage |
| Prefill throughput | No regression vs baseline | Pre-rotation adds one matmul |
| Test coverage | All existing tests + fused path tests | Regression protection |

### Expected Performance (RTX 4090)

| Metric | Current Unfused | Projected Fused | Improvement |
|--------|----------------|-----------------|-------------|
| Decode tokens/sec | 8.9 | 25-35 | 2.8-3.9x faster |
| Memory per layer (decode step) | ~25 MB read/write | ~3 MB read | 8x less traffic |
| Overhead vs baseline | 3.36x slower | 0.85-1.1x (near parity) | Overhead eliminated |
| KV cache size | 435 MiB | 435 MiB | Same (storage unchanged) |

_Confidence: MEDIUM — speedup projection from roofline analysis, not measured. Actual numbers depend on kernel tuning, occupancy, and Triton compiler efficiency on SM89._

### Prerequisites and Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| Triton >= 3.0 | Available in .venv (via torch) | Check `triton.__version__` |
| Dejan.ai code archive | Available online | Download `turboquant.zip` |
| Molmo2-4B weights | Cached locally | Already used in Experiments 001-004 |
| RTX 4090 GPU | Available | All prior experiments validated |
| HuggingFace `AttentionInterface` | Requires transformers >= 4.50 | Verify version |

### File Structure (Proposed)

```
turboquant-consumer/
  src/turboquant_consumer/
    triton/                      # NEW: Triton kernel module
      __init__.py
      fused_qk_attention.py      # Triton kernel (vendored + adapted from Dejan.ai)
      molmo2_integration.py      # AttentionInterface registration + Molmo2 patching
    kv_cache.py                  # Existing (unchanged)
    compressors.py               # Existing (unchanged)
    ...
  tests/
    test_triton_attention.py     # NEW: Kernel correctness tests
    test_molmo2_integration.py   # NEW: End-to-end integration tests
```

---

## Research Conclusion

### Verdict: Build It (Vendor + Adapt)

The fused TurboQuant Triton kernel is the highest-impact next step for the turboquant-consumer project. The Dejan.ai kernel provides a proven starting point — GQA-aware, fused, tested on RTX 4090 — and the adaptation for Molmo2 is a bounded 2-3 day engineering effort, not a research project.

### What This Unlocks

With the fused kernel, the complete TurboQuant story for Molmo2 becomes:

| Capability | Before Kernel | After Kernel |
|-----------|--------------|-------------|
| KV cache compression | 3.76x (TQ4 nibble) | 3.76x (same storage) |
| Decode throughput | 8.9 tok/s (3.36x slower) | 25-35 tok/s (near baseline) |
| Practical usability | Proof-of-concept only | **Production-viable** |
| Claim | "First TurboQuant on Molmo2" | "First TurboQuant on Molmo2 **with fused attention**" |

### What We Don't Need

- **Custom CUDA kernels** — Triton is sufficient; BitDecoding's CUDA/PTX approach is overkill for our scale
- **Flash Attention fusion** — The Q@K^T-only kernel (Dejan.ai style) is enough for decode. Full Flash fusion is Phase 2.
- **3-bit packing** — TQ4 nibble at 3.76x is the sweet spot. 3-bit crossing byte boundaries adds complexity for 30% more compression.
- **vLLM integration now** — HuggingFace `AttentionInterface` is the right integration point. vLLM can come later.

### Sources

- [Dejan.ai TurboQuant Triton kernel](https://dejan.ai/blog/turboquant/) — Primary kernel source, downloadable code archive
- [KIVI](https://github.com/jy-yuan/KIVI) — GQA + INT2/4 fused dequant reference
- [BitDecoding](https://github.com/DD-DuDa/BitDecoding) — CUDA-only, 7.5x speedup, design patterns
- [Kitty](https://github.com/Summer-Summer/Kitty) — Mixed-precision 2-bit KV cache
- [Anatomy of Triton Attention Kernel](https://arxiv.org/abs/2511.11581) — Triton GQA attention design patterns
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — CUDA FP4/FP8 fused attention
- [HuggingFace Attention Backends](https://huggingface.co/docs/transformers/main/attention_interface) — `AttentionInterface.register()` API
- [PyTorch triton_op](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — Triton kernel registration
- [vLLM Backend Registry](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py) — Plugin API
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — arXiv 2504.19874, ICLR 2026
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Official TurboQuant announcement
- Molmo2 `modeling_molmo2.py` — Local HuggingFace cache analysis
- RTX 4090 Ada Lovelace architecture specifications
