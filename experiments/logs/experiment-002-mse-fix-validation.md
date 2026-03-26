## Experiment: 002 — MSE-only fix validation (text + video, Molmo2-4B)

**Date:** 2026-03-25
**Hardware:** RTX 4090 (24 GB), AMD 7800X3D, 128 GB DDR5
**Model:** allenai/Molmo2-4B (bfloat16, device_map="auto")
**Baseline Config:** Standard DynamicCache, no compression
**Experimental Config:** TurboQuantKVCache with TurboQuantCompressorMSE for BOTH K and V (3-bit)

### Hypothesis

Switching from TurboQuantProd (2-bit MSE + 1-bit QJL) to TurboQuantMSE (full 3-bit
MSE) for the key compressor should fix the garbled output from Experiment 001.
The QJL correction is invisible to standard Q@K^T attention in the drop-in cache,
so we were wasting 1 bit of MSE resolution.

### Setup

Root cause from party mode review:
- Drop-in DynamicCache wrapper uses standard attention (Q @ K^T on decompressed keys)
- TurboQuantProd's QJL correction only benefits `estimate_inner_product()` (custom kernel)
- 2-bit MSE reconstruction (~87% cosine sim) compounded over 36 layers = garbled output
- 3-bit MSE reconstruction (~95% cosine sim) should be sufficient

Fix: one-line change in `kv_cache.py` — swap `TurboQuantCompressorV2` to
`TurboQuantCompressorMSE` for key_compressor.

### Results

**Run 002: Text-only**

| Metric | Baseline | TQ3 MSE | Delta |
|--------|----------|---------|-------|
| Output text | "Four" | "Four" | **IDENTICAL** |
| Tokens/sec | 4.9 | 1.6 | 3x slower |
| VRAM peak | 10,006 MiB | 10,006 MiB | Same |

**Run 003: Video (Seinfeld clip01.mp4)**

| Metric | Baseline | TQ3 MSE | Delta |
|--------|----------|---------|-------|
| Input tokens | 11,387 | 11,387 | Same |
| Output tokens | 256 | 256 | Same |
| Tokens/sec | 28.5 | 21.8 | 1.3x slower (23% overhead) |
| VRAM peak | 18,044 MiB | 18,044 MiB | Same (accuracy-only) |
| Output quality | Coherent scene description | Coherent scene description | Different details, both valid |

### Observations

1. **Text-only: IDENTICAL output.** The MSE fix completely resolves the garbled text
   from Experiment 001. At 3-bit MSE, the reconstruction fidelity is sufficient for
   autoregressive generation across 36 layers.

2. **Video: coherent but different.** Both baseline and TQ3 produce valid descriptions
   of the same Seinfeld scene, but focus on different visual details. This is expected —
   quantization noise shifts attention slightly, leading to different but equally valid
   caption paths. This is consistent with the paper's "zero accuracy loss" claim
   (task-level quality preserved, not token-level identity).

3. **Overhead scales well.** Text-only showed 3x slowdown (dominated by per-token
   quantize/dequantize overhead on 2 tokens). Video with 11K input tokens and 256 output
   tokens showed only 1.3x slowdown — the overhead amortizes over longer sequences.

4. **VRAM unchanged.** Expected — the drop-in wrapper compresses then decompresses,
   storing lossy FP32 in the standard DynamicCache. No actual memory savings.

### Next Steps

1. **Commit the MSE fix and benchmark harness** — the core implementation is validated.

2. **Test with Molmo2-8B** — need to figure out OOM issue (model too large for
   device_map="auto" on 24 GB with bf16). Options:
   - Use 4-bit weight quantization (bitsandbytes) to fit 8B in ~6 GB weights
   - Use `device_map="cuda:0"` with `torch_dtype=torch.float16` instead of bf16

3. **Longer video clips** — try soup-nazi clips for more frames, stress-test the
   KV cache with more visual tokens.

4. **Phase 2 planning** — for actual VRAM savings, need to store CompressedValues
   directly in cache and dequantize on read. Or wait for vLLM native support.
