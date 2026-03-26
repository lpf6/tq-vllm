# TurboQuant Consumer — Roadmap

Implementation status and path forward for TurboQuant KV cache compression
on consumer GPUs (RTX 4090, 24 GB VRAM) with Molmo2 vision-language models.

**Paper:** arXiv 2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

---

## Completed

### Layer 1: Core Quantization Algorithm

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Lloyd-Max codebook solver | Done | 8 | Gaussian approx for d >= 64, `@lru_cache`d |
| `TurboQuantMSE` (Stage 1) | Done | 6 | Rotation + scalar quantize, ~95% cosine sim at 3-bit |
| `TurboQuantProd` (Stage 2) | Done | 5 | MSE + QJL correction, unbiased inner products |

### Layer 2a: Production Compressors

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `CompressedKeys` / `CompressedValues` | Done | — | Dataclass containers |
| `TurboQuantCompressorV2` (keys) | Done | 6 | Includes `asymmetric_attention_scores()` |
| `TurboQuantCompressorMSE` (values) | Done | 2 | MSE-only for value reconstruction |

### Layer 2b: Compressed KV Cache (real VRAM savings)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| `TurboQuantKVCache` (accuracy-only) | Done | 7 | Compress-decompress round-trip, no VRAM savings |
| `CompressedDynamicCache` | Done | 13 | uint8 indices + fp32 norms, 1.94x compression |
| Benchmark harness (`--compressed`) | Done | — | A/B testing with Molmo2, VRAM + quality metrics |

### Experiments

| # | Date | Result | Key Finding |
|---|------|--------|-------------|
| 001 | 2026-03-25 | Failed | TurboQuantProd (2-bit MSE + 1-bit QJL) garbled output — QJL wasted in drop-in mode |
| 002 | 2026-03-25 | Passed | MSE-only fix: identical text output, coherent video (1.3x overhead) |
| 003 | 2026-03-25 | Passed | CompressedDynamicCache: coherent output, 1.94x compression, fp16 norms bug found and fixed |

---

## In Progress / Next Up

### P1: Long-sequence regression test

**Goal:** Catch precision bugs like the fp16 norms issue in CI, without needing a GPU or real model.

**Approach:** Synthetic test that creates a multi-layer cache (e.g., 36 layers, 1000+ tokens), compresses/decompresses, and verifies that the accumulated error doesn't exceed a threshold. Compare `CompressedDynamicCache` output vs `TurboQuantKVCache` output across many layers.

### P1: Molmo2-8B validation

**Goal:** Confirm CompressedDynamicCache works with the larger model. The 8B model recognizes character names (e.g., "Elaine", "Kramer") which 4B cannot.

**Approach:** Run benchmark with `--model allenai/Molmo2-8B --compressed`. May need `bitsandbytes` 4-bit weight quantization to fit model + compressed cache in 24 GB.

**Expected:** With ~800 MiB KV cache savings, 8B should have more headroom for context.

---

## Future Work

### P2: Index packing (higher compression)

**Goal:** Pack 3-bit indices more densely. Currently stored as uint8 (1 byte per index, 62.5% wasted bits).

| Packing | Bytes per 128 indices | Compression vs FP16 |
|---------|----------------------|---------------------|
| uint8 (current) | 128 | 1.94x |
| 4-bit nibbles | 64 | 3.76x |
| 3-bit packed | 48 | 4.74x |

**Approach:** Naive Python first (bit-shift packing/unpacking), then Triton kernel if the naive path is a bottleneck.

### P3: Triton fused dequant-attention kernel

**Goal:** Fuse dequantization with attention computation to avoid materializing full decompressed tensors.

**Why:** Current `CompressedDynamicCache` dequantizes the ENTIRE cache at every layer at every generation step (11K+ vectors through a 128x128 matrix multiply). This is the source of the 2.35x overhead. A fused kernel would:
1. Read uint8 indices + fp32 norms directly
2. Compute centroid lookup + rotation + scaling inline
3. Compute Q @ K^T without materializing the full K tensor
4. Reduce both latency AND peak VRAM

**Complexity:** Medium-high. Requires Triton kernel development and correctness validation.

### P4: vLLM native integration

**Goal:** Run TurboQuant as a vLLM KV cache backend, enabling compressed caching in production serving (not just HF transformers benchmarks).

**Status:** No official Google code released. vLLM integration expected Q2-Q3 2026 per community tracking. Once available, the molmo-video-analyzer pipeline would swap `--kv-cache-dtype fp8` for the TurboQuant option.

**Our contribution path:** The `turboquant-consumer` codebase could serve as a reference implementation or direct integration PR once vLLM exposes the cache backend API.

---

## Hardware Context

| Component | Spec | Relevance |
|-----------|------|-----------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) | All benchmarks run here |
| CPU | AMD 7800X3D | Codebook solving, data loading |
| RAM | 128 GB DDR5 | Model offloading when needed |
| Target model | Molmo2-4B / 8B | Vision-language model for video analysis |
| Target workload | Seinfeld clip analysis | 11K+ visual tokens at 2fps |
| Production stack | vLLM in Podman (CDI GPU) | Currently FP8 KV cache |

---

## Key Lessons

1. **FP16 norms are a trap.** At 10K+ token sequences across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32 for norms.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys. QJL correction only helps with `estimate_inner_product()` (custom kernel). Using QJL in drop-in mode wastes 1 bit of MSE resolution for nothing.

3. **Peak VRAM != KV cache size.** On Molmo2-4B with 11K tokens, forward-pass activations dominate peak VRAM (~90%). KV cache compression savings are real but invisible to `max_memory_allocated()`. They matter for max_model_len budgeting, not peak measurement.

4. **PyTorch treats uint8 as boolean masks.** Fancy indexing with uint8 tensors triggers boolean masking, not integer indexing. Always cast to `.long()` before centroid lookup.
