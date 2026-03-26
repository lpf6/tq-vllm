# TurboQuant Consumer

Implementation of Google's **TurboQuant** algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) for compressing transformer KV caches on consumer GPUs. Validated on **Molmo2 vision-language models** with real video inference on an RTX 4090.

## Headline Results

**3.76x KV cache compression with near-identical output quality** on Molmo2-4B processing 11K-token Seinfeld video clips:

| Mode | KV Cache | Compression | Output Quality | Overhead |
|------|----------|-------------|----------------|----------|
| FP16 baseline | 1,639 MiB | 1.0x | -- | -- |
| TQ3 (3-bit uint8) | 845 MiB | 1.94x | Coherent, different details | 2.35x slower |
| **TQ4 (4-bit nibble)** | **435 MiB** | **3.76x** | **Near-identical (100+ tokens match)** | 3.36x slower |

> First TurboQuant implementation validated on a vision-language model (VLM) with video input.

## What's Here

- **Core algorithm** -- Lloyd-Max codebook solver, TurboQuantMSE (Stage 1), TurboQuantProd (Stage 2 with QJL correction)
- **CompressedDynamicCache** -- Drop-in KV cache wrapper storing uint8 indices + fp32 norms with lazy dequantization. At `bits=4`, indices are nibble-packed (two per byte) for 3.76x compression.
- **Benchmark harness** -- A/B testing CLI comparing baseline vs compressed on any HuggingFace model
- **62 tests** -- Including long-sequence regression tests (36 layers, 1024 tokens) that catch precision bugs

## Quickstart

```bash
# Install
git clone https://github.com/Alberto-Codes/turboquant-consumer.git
cd turboquant-consumer
uv sync

# Run tests
uv run pytest tests/ -v

# Benchmark on Molmo2-4B (requires GPU + model weights)
uv run python -m turboquant_consumer.benchmark \
    --model allenai/Molmo2-4B \
    --bits 4 --compressed \
    --video /path/to/video.mp4 \
    --max-new-tokens 256
```

## Usage

```python
from transformers import DynamicCache
from turboquant_consumer import CompressedDynamicCache

# Wrap any HuggingFace DynamicCache
cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)

# Pass cache (not the wrapper) to model.generate()
# Compression happens transparently on every cache.update()
```

## Key Findings

1. **FP16 norms are a trap.** At 10K+ tokens across 36 layers, fp16 norm precision loss compounds and flips low-confidence logits. Always use fp32.

2. **QJL is invisible in drop-in mode.** Standard attention does `Q @ K.T` on decompressed keys -- QJL correction only helps with a custom attention kernel. Using QJL wastes 1 bit of MSE resolution.

3. **TQ4 nibble beats TQ3 unpacked.** 4-bit with nibble packing gives 3.76x compression and ~97% cosine similarity. 3-bit unpacked gives only 1.94x at ~95%. Packing 3-bit indices across byte boundaries is hard and only 30% better.

4. **Peak VRAM is activation-dominated.** KV cache is ~9% of peak VRAM during prefill. Compression savings are real in permanent storage but invisible to `max_memory_allocated()`.

## Hardware Tested

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4090 (24 GB GDDR6X) |
| CPU | AMD 7800X3D |
| RAM | 128 GB DDR5 |
| Model | Molmo2-4B (bfloat16) |
| Workload | Seinfeld clips, ~11K visual tokens at 2fps |

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) -- Module map, dependency DAG, data flow diagrams, design decisions
- [`docs/ROADMAP.md`](docs/ROADMAP.md) -- Implementation status, next steps, key lessons
- [`experiments/logs/`](experiments/logs/) -- All 4 experiment logs with full results

## Status

**Pre-alpha.** The core algorithm and compressed cache are validated. The current bottleneck is decode throughput (3.36x overhead from full-cache dequantization every step). A [fused Triton attention kernel](docs/ROADMAP.md) would eliminate this overhead -- research and implementation plan are complete.

## Reference

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Han, Insu and Daliri, Majid and Karbasi, Amin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License

MIT
