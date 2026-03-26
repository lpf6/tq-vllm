## Experiment: 003 — CompressedDynamicCache VRAM validation (Molmo2-4B + video)

**Date:** 2026-03-25
**Hardware:** RTX 4090 (24 GB), AMD 7800X3D, 128 GB DDR5
**Model:** allenai/Molmo2-4B (bfloat16, device_map="auto")
**Baseline Config:** Standard DynamicCache, no compression
**Experimental Config:** CompressedDynamicCache with uint8 indices + float32 norms (3-bit)

### Hypothesis

CompressedDynamicCache (Layer 2b) should produce coherent output comparable to
the accuracy-only TurboQuantKVCache, while storing the KV cache in ~2x less memory
(uint8 indices + float32 norms vs full FP16 tensors).

### Setup

- Video: Seinfeld clip01.mp4 (~11K visual tokens at 2fps)
- Prompt: "Describe what happens in this video scene in detail..."
- Generation: 256 tokens, greedy decoding (do_sample=False)
- Benchmark: baseline run first, then compressed run
- VRAM measured via torch.cuda.max_memory_allocated() after reset

### Results

| Metric | Baseline | Compressed TQ3 | Delta |
|--------|----------|-----------------|-------|
| Input tokens | 11,397 | 11,397 | Same |
| Output tokens | 256 | 256 | Same |
| Tokens/sec | 30.5 | 13.0 | 2.35x slower |
| VRAM peak | 18,058 MiB | 18,057 MiB | ~0 (see analysis) |
| Output quality | Coherent Seinfeld description | Coherent Seinfeld description | Different details, both valid |

**KV Cache Compression Stats:**

| Metric | Value |
|--------|-------|
| Compressed KV cache | 844.9 MiB |
| Baseline KV cache (FP16 equivalent) | 1,638.6 MiB |
| Compression ratio | 1.94x |
| Savings (theoretical) | 793.7 MiB |

### Critical Bug Found and Fixed: FP16 Norms

Initial implementation stored norms as float16. This caused **garbled output**
("In the video,1.0 0 0 0 0...") for sequences ≥ ~11,400 tokens but worked fine
for shorter sequences (~11,385 tokens).

**Root cause:** float16 norm precision loss (~0.01% per vector) accumulated across
36 transformer layers, shifting attention logits at low-confidence decision points.
Token-by-token logit analysis showed the first 4 tokens matched the accuracy-only
path, then diverged at step 5 where the logit margin was < 0.5.

**Fix:** Store norms as float32 instead of float16. Cost: 2 extra bytes per token
per head (132 vs 130 bytes). Compression ratio drops negligibly from 1.97x to 1.94x.

### Observations

1. **Output quality: coherent.** Both baseline and compressed mode produce valid
   descriptions of the Seinfeld scene. Different details (expected — quantization
   noise shifts attention), both correct.

2. **VRAM peak unchanged.** The ~794 MiB theoretical savings do NOT show up in
   `torch.cuda.max_memory_allocated()`. The peak is dominated by model forward-pass
   activations during prefill (11K tokens × 32 attention heads), not the KV cache.
   The KV cache is only ~9% of post-model VRAM. The savings are REAL in permanent
   storage but invisible to peak measurement.

3. **2.35x overhead.** Worse than accuracy-only (1.3x in Experiment 002). The
   difference: compressed mode dequantizes the FULL cache (all 11K+ tokens) at
   every layer at every generation step. Accuracy-only only compresses/decompresses
   the NEW token. A Triton kernel for fused dequant-attention would eliminate this.

4. **FP16 norms are a trap.** The 2-byte savings per vector are not worth the
   output degradation at long contexts. Float32 norms are required for correctness.

### Next Steps

1. **Measure VRAM after generation (not peak)** — Use `torch.cuda.memory_allocated()`
   at the END of generation, not `max_memory_allocated()`. The post-generation
   cache size is where compression savings are visible.

2. **Triton dequant-attention kernel** — Fuse dequantization with attention to avoid
   materializing full decompressed tensors. Would reduce both overhead and peak VRAM.

3. **Index packing** — Pack two 3-bit indices into 6 bits (or four into 12 bits) to
   approach the theoretical 5-6x compression from the paper.

4. **Molmo2-8B validation** — With ~800 MiB KV savings, 8B model may fit more
   comfortably. Try with bitsandbytes 4-bit weight quantization.
