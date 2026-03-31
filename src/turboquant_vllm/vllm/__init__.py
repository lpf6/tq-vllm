"""TQ4 compressed KV cache backend for vLLM.

Registers custom attention backends that store KV cache pages in
TurboQuant 4-bit format (68 bytes/token/head vs 256 bytes FP16 = 3.76x
compression).

Three backends are available:
1. TQ4AttentionBackend (FlashAttention-based): For GPUs with compute capability 8.0+
2. TQ4TritonBackend (TritonAttention-based): For GPUs with compute capability 7.5+
   (e.g., RTX 2080 Ti)
3. TQ4FlashInferBackend (FlashInfer-based): For GPUs with compute capability 7.5+
   (e.g., RTX 2080 Ti, RTX 3090, A100, H100)

Attributes:
    TQ4AttentionBackend: FlashAttention-based backend.
    TQ4AttentionImpl: Attention implementation (FlashAttention-based).
    TQ4TritonBackend: Triton-based backend.
    TQ4TritonImpl: Triton-based attention implementation.
    TQ4FlashInferBackend: FlashInfer-based backend.
    TQ4FlashInferImpl: FlashInfer-based attention implementation.
    register_tq4_backend: Callable to register TQ4 backend (uses TQ4_BACKEND env var).
    register_tq4_triton_backend: Callable to register the Triton backend directly.
    register_tq4_flashinfer_backend: Callable to register the FlashInfer backend directly.

See Also:
    :mod:`turboquant_vllm.kv_cache`: CompressedDynamicCache for HF transformers.

Usage:
    The backend registers automatically via the ``vllm.general_plugins``
    entry point when turboquant-vllm is installed with the ``vllm``
    extra::

        pip install turboquant-vllm[vllm]

    Use the ``TQ4_BACKEND`` environment variable to select the backend::

        # For GPUs with compute capability 8.0+ (e.g., A100, RTX 3090)
        export TQ4_BACKEND=FA
        vllm serve <model> --attention-backend CUSTOM

        # For GPUs with compute capability 7.5+ using Triton (e.g., RTX 2080 Ti)
        export TQ4_BACKEND=TRITON
        vllm serve <model> --attention-backend CUSTOM

        # For GPUs with compute capability 7.5+ using FlashInfer (e.g., RTX 2080 Ti)
        export TQ4_BACKEND=FLASHINFER
        vllm serve <model> --attention-backend CUSTOM

    Or register manually before starting vLLM::

        # Using environment variable (recommended)
        import os
        os.environ["TQ4_BACKEND"] = "TRITON"  # or "FA", "FLASHINFER"
        from turboquant_vllm.vllm import register_tq4_backend
        register_tq4_backend()

        # Or register specific backend directly
        from turboquant_vllm.vllm import register_tq4_triton_backend
        register_tq4_triton_backend()
"""

from turboquant_vllm.vllm.tq4_backend import (
    TQ4AttentionBackend,
    TQ4AttentionImpl,
    TQ4FullAttentionSpec,
    register_tq4_backend,
)
from turboquant_vllm.vllm.tq4_triton_backend import (
    TQ4TritonBackend,
    TQ4TritonImpl,
    register_tq4_triton_backend,
)
from turboquant_vllm.vllm.tq4_flashinfer_backend import (
    TQ4FlashInferBackend,
    TQ4FlashInferImpl,
    register_tq4_flashinfer_backend,
)

__all__ = [
    "TQ4AttentionBackend",
    "TQ4AttentionImpl",
    "TQ4FullAttentionSpec",
    "register_tq4_backend",
    "TQ4TritonBackend",
    "TQ4TritonImpl",
    "register_tq4_triton_backend",
    "TQ4FlashInferBackend",
    "TQ4FlashInferImpl",
    "register_tq4_flashinfer_backend",
]
