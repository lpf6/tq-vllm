## Experiment: 001 — Initial harness validation (text-only, Molmo2-4B)

**Date:** 2026-03-25
**Hardware:** RTX 4090 (24 GB), AMD 7800X3D, 128 GB DDR5
**Model:** allenai/Molmo2-4B (bfloat16, device_map="auto")
**Baseline Config:** Standard DynamicCache, no compression
**Experimental Config:** TurboQuantKVCache wrapper, 3-bit K+V compression (TQ3)

### Hypothesis

TurboQuant 3-bit K+V compression should produce similar output quality to the
uncompressed baseline for a simple text prompt, with measurable but acceptable
quality degradation. VRAM tracking should capture peak memory during generation.

### Setup

```bash
uv run python -m turboquant_consumer.benchmark \
    --model allenai/Molmo2-4B \
    --prompt "What is 2+2? Answer in one word." \
    --max-new-tokens 32
```

Model loaded with `device_map="auto"` (accelerate). Transformers 4.57.6.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Model config: 36 layers, 32 attention heads, 8 KV heads (GQA), head_dim=128.

### Results

| Metric | Baseline | TQ3 K+V | Delta |
|--------|----------|---------|-------|
| Output text | "Four" | "The number 222 is is a number..." (garbled) | FAIL |
| Output tokens | 2 | 32 (hit max) | +30 |
| Tokens/sec | 2.7 | 3.0 | +11% |
| Time | 0.7s | 10.8s | +14.6x |
| VRAM peak | 0 MiB (broken) | 0 MiB (broken) | N/A |

### Observations

1. **Harness works end-to-end** — both baseline and TurboQuant runs complete without crashes.

2. **TurboQuant output is garbled** — baseline correctly answers "Four", TQ3 produces
   repetitive incoherent text. This is a significant quality issue at 3-bit that needs
   investigation. Possible causes:
   - The DynamicCache monkey-patch may interact poorly with Molmo2's GQA attention
     (8 KV heads vs 32 query heads). Our compressor initializes with head_dim=128
     but the cache.update() receives tensors shaped (batch, 8, seq, 128) not
     (batch, 32, seq, 128). The compressor should handle this fine since it operates
     on the last dim, but GQA may change the attention score dynamics.
   - 3-bit TurboQuantProd uses only 2 bits for MSE + 1 bit QJL. At 2-bit MSE,
     reconstruction cosine similarity is ~0.87 (measured in tests). This may be
     too lossy for autoregressive generation where errors compound across tokens.
   - The compress→decompress→store pattern means quantization noise is injected
     into every layer's KV cache every token step. Over 36 layers and multiple
     decode steps, this compounds.

3. **VRAM tracking is broken** — shows 0 MiB because `device_map="auto"` offloaded
   the model partially to CPU. `torch.cuda.max_memory_allocated()` doesn't track
   CPU-offloaded tensors. Need to either force full GPU placement or use
   `nvidia-smi` polling instead.

4. **TQ run 14.6x slower** — mostly because quantize/dequantize runs in Python on
   CPU-offloaded tensors. With full GPU placement this would be much closer.

5. **Baseline is very fast** — 2.7 tok/s for a short prompt on a partially-offloaded
   model is reasonable. The 0.7s for 2 tokens includes KV cache prefill.

### Next Steps

1. **Investigate TQ3 quality degradation** — Try TQ4 (3-bit MSE + 1 bit QJL) to see
   if the extra MSE bit fixes generation quality. If TQ4 works and TQ3 doesn't, the
   issue is reconstruction fidelity, not the integration.

2. **Fix VRAM tracking** — Force `device_map="cuda:0"` instead of `"auto"` for 4B
   model (fits in 24 GB). This gives accurate VRAM measurements.

3. **Try a video input** — Use one of the Seinfeld clips to test with real visual
   tokens (which stress the KV cache more).

4. **Profile the hot path** — Measure time spent in quantize vs dequantize vs
   attention vs everything else to understand the 14.6x slowdown breakdown.
