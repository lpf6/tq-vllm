"""Fused Triton kernels for TurboQuant compressed attention.

Provides a fused kernel that computes Q @ compressed_K^T directly from
nibble-packed 4-bit indices, avoiding full key dequantization.

Attributes:
    fused_qk_scores: Compute attention scores from pre-rotated queries
        and nibble-packed compressed keys.

Examples:
    ```python
    from turboquant_consumer.triton import fused_qk_scores

    scores = fused_qk_scores(
        q_rotated,
        packed_indices,
        norms,
        centroids,
        scale,
        n_q_heads=32,
        n_kv_heads=8,
    )
    ```

See Also:
    :mod:`turboquant_consumer.kv_cache`: CompressedDynamicCache storage layer.
"""

from turboquant_consumer.triton.fused_qk_attention import fused_qk_scores

__all__ = ["fused_qk_scores"]
