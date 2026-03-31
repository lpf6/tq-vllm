"""TQ4 compressed KV cache attention backend for vLLM using FlashInfer.

This backend extends FlashInferBackend to support TQ4 (4-bit TurboQuant)
compressed KV cache format. It provides 3.76x compression compared to FP16.

The implementation follows the same pattern as the TritonAttention-based TQ4 backend
but adapts it for the FlashInferBackend which supports compute capability 7.5+.

Usage:
    Set environment variable before starting vLLM:
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    
    Or use vLLM's attention_backend parameter:
    --attention-backend FLASHINFER
    
    Then register the TQ4 backend:
    
        from turboquant_vllm.vllm import register_tq4_flashinfer_backend
        register_tq4_flashinfer_backend()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadataBuilder,
    FlashInferMetadata,
    FIPrefill,
    FIDecode,
    TRTLLMPrefill,
    TRTLLMDecode,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from turboquant_vllm.quantizer import TurboQuantMSE
from turboquant_vllm.triton.tq4_compress import tq4_compress
from turboquant_vllm.triton.tq4_decompress import tq4_decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionImplBase,
        AttentionMetadataBuilder,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TQ4 constants
# ---------------------------------------------------------------------------

TQ4_BITS = 4
TQ4_SEED = 42
TQ4_NORM_BYTES = 4  # fp32


def _tq4_bytes_per_token(head_dim: int) -> int:
    """Packed byte count for one token, one KV head, one of K or V.
    
    Returns:
        Byte count: head_dim // 2 (nibble-packed indices) + 4 (fp32 norm).
    """
    return head_dim // 2 + TQ4_NORM_BYTES


def _tq4_bytes_per_token_kv(head_dim: int) -> int:
    """Total packed bytes per token per KV head (K + V combined)."""
    return 2 * _tq4_bytes_per_token(head_dim)


# ---------------------------------------------------------------------------
# KV cache spec with TQ4 packed page size
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class TQ4FullAttentionSpec(FullAttentionSpec):
    """KV cache spec with TQ4 packed page size.
    
    Overrides real_page_size_bytes so the block allocator provisions
    buffers sized for the packed TQ4 format (3.76x smaller than FP16).
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        return (
            self.block_size
            * self.num_kv_heads
            * _tq4_bytes_per_token_kv(self.head_size)
        )


# ---------------------------------------------------------------------------
# TQ4 FlashInfer Backend
# ---------------------------------------------------------------------------

class TQ4FlashInferMetadataBuilder(FlashInferMetadataBuilder):
    """Metadata builder for TQ4 with FlashInfer.
    
    Inherits all metadata-building logic from FlashInferMetadataBuilder.
    CUDA graph support follows the same pattern as the base FlashInfer backend.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: object,
        kv_cache_spec: object,
    ) -> AttentionCGSupport:
        """Report CUDA graph support based on FlashInfer's capability."""
        # Delegate to parent class logic
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)


class TQ4FlashInferBackend(FlashInferBackend):
    """TQ4 compressed KV cache attention backend using FlashInfer.
    
    This backend provides TQ4 quantization support for GPUs with compute
    capability 7.5+ (e.g., RTX 2080 Ti, RTX 3090, A100, H100).
    
    The cache stores nibble-packed TQ4 indices + fp32 norms as raw bytes.
    get_kv_cache_shape() returns a 3D (NB, BS, bytes_per_token) layout
    matching the packed format.
    """

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        """Return backend name."""
        return "TQ4_FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        """Return TQ4FlashInferImpl."""
        return TQ4FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        """Return TQ4FlashInferMetadataBuilder for CUDA graph support."""
        return TQ4FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Packed TQ4 cache: (num_blocks, block_size, total_bytes).
        
        The last dimension packs K and V data for all heads as raw bytes:
        [K_indices | K_norms | V_indices | V_norms].
        """
        total_bytes = num_kv_heads * _tq4_bytes_per_token_kv(head_size)
        return (num_blocks, block_size, total_bytes)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Raise to trigger identity fallback in reshape.
        
        The inherited FlashInferBackend returns a 5-element stride
        order for the standard (2, NB, BS, H, D) shape. Our 3D packed
        layout (NB, BS, total_bytes) needs identity ordering.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TQ4 FlashInfer Implementation
# ---------------------------------------------------------------------------

class TQ4FlashInferImpl(FlashInferImpl):
    """TQ4 attention implementation using FlashInfer backend.
    
    Stores packed TQ4 bytes in a uint8 cache for real VRAM savings.
    Each forward() call:
    
    1. Compresses incoming K/V tokens to TQ4 packed bytes.
    2. Scatter-writes packed bytes to the uint8 cache via slot_mapping.
    3. Decompresses the relevant cache blocks to FP16 for FlashInfer.
    4. Calls FlashInfer's prefill/decode wrappers with the FP16 data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize TQ4 attention with compression primitives."""
        # Initialize FlashInferImpl first (it handles attn_type validation)
        super().__init__(*args, **kwargs)

        # Use attributes set by super().__init__()
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        # TQ4 compression primitives (deterministic from seed, shared across layers)
        quantizer = TurboQuantMSE(head_size, TQ4_BITS, seed=TQ4_SEED)

        # Eagerly move primitives to the target device
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        device = (
            vllm_config.device_config.device
            if vllm_config is not None
            else torch.device("cpu")
        )

        self._tq4_rotation = quantizer.rotation.to(device)  # (D, D) fp32
        self._tq4_centroids = quantizer.codebook.centroids.to(device)  # (16,) fp32
        self._tq4_boundaries = quantizer.codebook.boundaries.to(device)  # (15,) fp32
        # Pre-split rotation.T for fused compress kernel (contiguous loads)
        rot_t = quantizer.rotation.T.contiguous()
        self._tq4_rot_T_even = rot_t[:, 0::2].contiguous().to(device)  # (D, D//2) fp32
        self._tq4_rot_T_odd = rot_t[:, 1::2].contiguous().to(device)  # (D, D//2) fp32

        # Byte layout offsets within the last dimension of the packed cache.
        # Layout: [K_indices(H*D/2) | K_norms(H*4) | V_indices(H*D/2) | V_norms(H*4)]
        half_D = head_size // 2
        self._half_D = half_D
        self._k_idx_end = num_kv_heads * half_D
        self._k_norm_end = self._k_idx_end + num_kv_heads * TQ4_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * half_D
        self._total_bytes = self._v_idx_end + num_kv_heads * TQ4_NORM_BYTES

        # CUDA graph scratch buffers - lazy-allocated on first forward()
        self._cg_buffers_ready = False

        # Buffer bounds from config
        self._max_prefill_len = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if vllm_config is not None
            else 2048
        )
        self._max_model_len = (
            vllm_config.model_config.max_model_len if vllm_config is not None else 6144
        )

        logger.info(
            "TQ4FlashInferImpl: %d KV heads, head_size=%d, "
            "%d bytes/token (%.2fx compression vs FP16)",
            num_kv_heads,
            head_size,
            self._total_bytes,
            (2 * num_kv_heads * head_size * 2) / self._total_bytes,
        )

    def _init_cg_buffers(
        self, kv_cache: torch.Tensor, compute_dtype: torch.dtype
    ) -> None:
        """Pre-allocate CUDA graph scratch buffers from kv_cache shape.
        
        Called once during vLLM warmup (first forward), before CUDA graph
        capture. Uses max-size + slicing pattern.
        
        Args:
            kv_cache: (num_blocks, block_size, total_bytes) uint8 cache.
            compute_dtype: Model compute dtype (e.g. torch.bfloat16).
        """
        num_blocks, block_size, _ = kv_cache.shape
        max_tokens = num_blocks * block_size
        device = kv_cache.device
        H = self.num_kv_heads
        D = self.head_size

        # Decompress buffers: bounded by max_model_len
        decompress_tokens = min(self._max_model_len, max_tokens)
        self._cg_decompress_k = torch.empty(
            decompress_tokens, H, D, dtype=compute_dtype, device=device
        )
        self._cg_decompress_v = torch.empty_like(self._cg_decompress_k)

        # Prefill scratch buffers: bounded by max_prefill_len
        prefill_tokens = min(self._max_prefill_len, max_tokens)
        self._cg_prefill_k = torch.empty(
            prefill_tokens, H, D, dtype=compute_dtype, device=device
        )
        self._cg_prefill_v = torch.empty_like(self._cg_prefill_k)

        # Compress output buffers for one decode step (single token)
        half_D = self._half_D
        self._cg_compress_packed = torch.empty(
            1, H, half_D, dtype=torch.uint8, device=device
        )
        self._cg_compress_norms = torch.empty(
            1, H, 1, dtype=torch.float32, device=device
        )

        # Q rotation buffer for decode (single token, fp32 for precision)
        self._cg_q_rot = torch.empty(
            1, self.num_heads, D, dtype=torch.float32, device=device
        )

        # Q rotation cast buffer (compute dtype for attention input)
        self._cg_q_rot_cast = torch.empty(
            1, self.num_heads, D, dtype=compute_dtype, device=device
        )

        # Compress row assembly buffer for _compress_and_store
        self._cg_compress_row = torch.empty(
            1, self._total_bytes, dtype=torch.uint8, device=device
        )

        self._cg_buffers_ready = True
        logger.info(
            "TQ4 FlashInfer CUDA graph buffers allocated: decompress=%s "
            "(tokens=%d), prefill=%s (tokens=%d)",
            self._cg_decompress_k.shape,
            decompress_tokens,
            self._cg_prefill_k.shape,
            prefill_tokens,
        )

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        compress_out: tuple[torch.Tensor, torch.Tensor] | None = None,
        row_out: torch.Tensor | None = None,
    ) -> None:
        """Compress K/V and scatter-write TQ4 bytes to packed cache.
        
        Args:
            key: (N, H, D) new key tokens.
            value: (N, H, D) new value tokens.
            kv_cache: (NB, BS, total_bytes) uint8 packed cache.
            slot_mapping: (num_actual_tokens,) flat slot indices.
            compress_out: Optional pre-allocated (packed, norms) buffers.
            row_out: Optional pre-allocated row assembly buffer (N, total_bytes).
        """
        N = key.shape[0]
        H = self.num_kv_heads

        # Build packed byte row per token: [K_idx | K_norm | V_idx | V_norm]
        row = (
            row_out[:N]
            if row_out is not None
            else torch.empty(N, self._total_bytes, dtype=torch.uint8, device=key.device)
        )

        k_packed, k_norms = tq4_compress(
            key,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._tq4_boundaries,
            out=compress_out,
        )
        row[:, : self._k_idx_end] = k_packed.reshape(N, -1)
        row[:, self._k_idx_end : self._k_norm_end] = (
            k_norms.reshape(N, H).contiguous().view(torch.uint8)
        )

        v_packed, v_norms = tq4_compress(
            value,
            self._tq4_rot_T_even,
            self._tq4_rot_T_odd,
            self._tq4_boundaries,
            out=compress_out,
        )
        row[:, self._k_norm_end : self._v_idx_end] = v_packed.reshape(N, -1)
        row[:, self._v_idx_end :] = v_norms.reshape(N, H).contiguous().view(torch.uint8)

        # Scatter-write to flat cache using slot_mapping
        num_actual = slot_mapping.shape[0]
        flat_cache = kv_cache.view(-1, self._total_bytes)
        flat_cache[slot_mapping[:num_actual]] = row[:num_actual]

    def _decompress_cache_paged(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        compute_dtype: torch.dtype,
        *,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompress only the physical blocks referenced by block_table.
        
        Args:
            kv_cache: (NB, BS, total_bytes) uint8 packed cache.
            block_table: (batch, max_blocks_per_seq) int32 block table.
            seq_lens: (batch,) int32 sequence lengths.
            compute_dtype: Output dtype (e.g., torch.bfloat16).
            out_k: Pre-allocated (max_tokens, H, D) buffer for keys.
            out_v: Pre-allocated (max_tokens, H, D) buffer for values.
        
        Returns:
            (key_cache, value_cache, remapped_block_table) where
            key/value are (num_compact_blocks, BS, H, D) and
            remapped_block_table maps logical blocks to compact indices.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        half_D = self._half_D
        D = self.head_size

        # Extract valid block indices from block_table using seq_lens
        max_blocks_per_seq = block_table.shape[1]
        blocks_needed = (seq_lens + BS - 1) // BS  # ceil division
        col_idx = torch.arange(max_blocks_per_seq, device=block_table.device).unsqueeze(0)
        valid_mask = col_idx < blocks_needed.unsqueeze(1)
        valid_block_indices = block_table[valid_mask]

        unique_blocks = torch.unique(valid_block_indices, sorted=True)
        num_unique = unique_blocks.numel()

        # Capacity check
        max_blocks_capacity = out_k.shape[0] // BS
        if num_unique <= max_blocks_capacity:
            k_buf = out_k
            v_buf = out_v
        else:
            logger.warning(
                "Paged decompress: %d unique blocks exceed "
                "pre-allocated capacity (%d blocks), using dynamic fallback",
                num_unique,
                max_blocks_capacity,
            )
            fallback_tokens = num_unique * BS
            k_buf = torch.empty(
                fallback_tokens, H, D, dtype=compute_dtype, device=kv_cache.device
            )
            v_buf = torch.empty_like(k_buf)

        # Gather referenced blocks and decompress
        selected = kv_cache[unique_blocks]  # (num_unique, BS, total_bytes)
        flat = selected.reshape(num_unique * BS, self._total_bytes)

        k_packed = flat[:, : self._k_idx_end].contiguous().reshape(-1, H, half_D)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )
        v_packed = (
            flat[:, self._k_norm_end : self._v_idx_end]
            .contiguous()
            .reshape(-1, H, half_D)
        )
        v_norms = (
            flat[:, self._v_idx_end :]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H, 1)
        )

        # Slice output buffers to exact size needed
        k_out_slice = k_buf[: num_unique * BS]
        v_out_slice = v_buf[: num_unique * BS]

        key_out = tq4_decompress(
            k_packed, k_norms, self._tq4_centroids, compute_dtype, out=k_out_slice
        )
        value_out = tq4_decompress(
            v_packed, v_norms, self._tq4_centroids, compute_dtype, out=v_out_slice
        )

        key_cache = key_out.reshape(num_unique, BS, H, D)
        value_cache = value_out.reshape(num_unique, BS, H, D)

        # Build remapped block table: old physical -> compact 0..N-1
        remap = torch.zeros(NB, dtype=block_table.dtype, device=block_table.device)
        remap[unique_blocks] = torch.arange(
            num_unique, dtype=block_table.dtype, device=block_table.device
        )
        remapped_block_table = remap[block_table]

        return key_cache, value_cache, remapped_block_table

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        """TQ4 attention: compress -> store -> pre-rotate Q -> decompress -> FlashInfer -> post-rotate.
        
        The rotation is applied to Q before attention and to the output
        after, saving O(cache_len) matmuls per decode step.
        """
        assert output is not None

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported with TQ4 backend"
            )

        # Profiling mode
        if attn_metadata is None:
            output.zero_()
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Lazy-init CUDA graph buffers on first forward (during warmup)
        if not self._cg_buffers_ready and kv_cache is not None:
            self._init_cg_buffers(kv_cache, compute_dtype=query.dtype)

        # Compress and store new K/V tokens
        if kv_cache is not None and key is not None and value is not None:
            if self._cg_buffers_ready and num_actual_tokens == 1:
                # Decode path with pre-allocated buffers
                self._compress_and_store(
                    key,
                    value,
                    kv_cache,
                    attn_metadata.slot_mapping,
                    compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                    row_out=self._cg_compress_row,
                )
            else:
                # Prefill path
                self._compress_and_store(key, value, kv_cache, attn_metadata.slot_mapping)

        # Pre-rotate Q
        q_slice = query[:num_actual_tokens]
        if self._cg_buffers_ready and num_actual_tokens == 1:
            # Decode: use pre-allocated buffer
            q_rot_buf = self._cg_q_rot[:1]
            torch.matmul(q_slice.float(), self._tq4_rotation.T, out=q_rot_buf)
            self._cg_q_rot_cast[:1].copy_(q_rot_buf)
            q_rot = self._cg_q_rot_cast[:1]
        else:
            # Prefill
            q_rot = (q_slice.float() @ self._tq4_rotation.T).to(q_slice.dtype)

        # Decompress KV cache
        if kv_cache is not None:
            if self._cg_buffers_ready and num_actual_tokens == 1:
                # Decode path
                key_cache, value_cache, remapped_bt = self._decompress_cache_paged(
                    kv_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    query.dtype,
                    out_k=self._cg_decompress_k,
                    out_v=self._cg_decompress_v,
                )
            else:
                # Prefill path
                key_cache, value_cache, remapped_bt = self._decompress_cache_paged(
                    kv_cache,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    query.dtype,
                    out_k=self._cg_prefill_k,
                    out_v=self._cg_prefill_v,
                )
        else:
            # No cache (shouldn't happen for decoder attention)
            key_cache = key[:num_actual_tokens].unsqueeze(0)
            value_cache = value[:num_actual_tokens].unsqueeze(0)
            remapped_bt = attn_metadata.block_table

        # Prepare KV cache for FlashInfer (needs 5D shape)
        # FlashInfer expects: (num_blocks, 2, block_size, num_kv_heads, head_size)
        BS = key_cache.shape[1]
        H = self.num_kv_heads
        D = self.head_size
        
        # Stack K and V to create the expected shape
        kv_cache_fi = torch.stack([key_cache, value_cache], dim=1)  # (N, 2, BS, H, D)

        # Get stride order for FlashInfer
        stride_order = TQ4FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache_fi.permute(*stride_order)

        # Run FlashInfer attention with rotated Q and rotated KV
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Handle prefill
        if num_prefill_tokens > 0:
            prefill_query = q_rot[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            if isinstance(attn_metadata.prefill, FIPrefill):
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                
                # Run FlashInfer prefill
                prefill_wrapper.run(
                    prefill_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[num_decode_tokens:num_actual_tokens],
                )
            elif isinstance(attn_metadata.prefill, TRTLLMPrefill):
                # TRTLLM prefill path
                from flashinfer.prefill import trtllm_batch_context_with_kv_cache
                from vllm.utils.flashinfer import use_trtllm_attention
                
                workspace_buffer = torch.zeros(
                    256 * 1024 * 1024, dtype=torch.uint8, device=query.device
                )
                
                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=kv_cache_permute,
                    workspace_buffer=workspace_buffer,
                    block_tables=attn_metadata.prefill.block_tables,
                    seq_lens=attn_metadata.prefill.seq_lens,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.scale,
                    bmm2_scale=1.0,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    out=output[num_decode_tokens:num_actual_tokens],
                )

        # Handle decode
        if num_decode_tokens > 0:
            decode_query = q_rot[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            if isinstance(attn_metadata.decode, FIDecode):
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                
                # Run FlashInfer decode
                decode_wrapper.run(
                    decode_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[:num_decode_tokens],
                )
            elif isinstance(attn_metadata.decode, TRTLLMDecode):
                # TRTLLM decode path
                from flashinfer.decode import trtllm_batch_decode_with_kv_cache
                
                workspace_buffer = torch.zeros(
                    256 * 1024 * 1024, dtype=torch.uint8, device=query.device
                )
                
                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=kv_cache_permute,
                    workspace_buffer=workspace_buffer,
                    block_tables=attn_metadata.decode.block_tables,
                    seq_lens=attn_metadata.decode.seq_lens,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.scale,
                    bmm2_scale=1.0,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    out=output[:num_decode_tokens],
                )

        # Post-rotate output by Pi (undo rotation space)
        out_slice = output[:num_actual_tokens]
        output[:num_actual_tokens] = (out_slice.float() @ self._tq4_rotation).to(
            out_slice.dtype
        )

        return output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_original_get_kv_cache_spec = None


def register_tq4_flashinfer_backend() -> None:
    """Register TQ4 FlashInfer backend.
    
    In addition to registering the backend class, this monkey-patches
    Attention.get_kv_cache_spec so that decoder attention layers
    return TQ4FullAttentionSpec (with dtype=torch.uint8 and TQ4-sized
    pages) instead of the standard FullAttentionSpec.
    
    Called automatically by the vllm.general_plugins entry point,
    or manually before starting vLLM::
    
        from turboquant_vllm.vllm import register_tq4_flashinfer_backend
        
        register_tq4_flashinfer_backend()
        # then start vLLM with --attention-backend TQ4_FLASHINFER
    """
    global _original_get_kv_cache_spec

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "turboquant_vllm.vllm.tq4_flashinfer_backend.TQ4FlashInferBackend",
    )

    # Register TQ4FullAttentionSpec in the KV cache manager mapping
    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

    if TQ4FullAttentionSpec not in spec_manager_map:
        spec_manager_map[TQ4FullAttentionSpec] = spec_manager_map[FullAttentionSpec]

    # Monkey-patch Attention.get_kv_cache_spec to return TQ4 spec
    from vllm.model_executor.layers.attention.attention import Attention

    if _original_get_kv_cache_spec is None:
        _original_get_kv_cache_spec = Attention.get_kv_cache_spec

    def _tq4_get_kv_cache_spec(self, vllm_config):
        spec = _original_get_kv_cache_spec(self, vllm_config)
        if isinstance(spec, FullAttentionSpec) and not isinstance(
            spec, TQ4FullAttentionSpec
        ):
            kwargs = {f.name: getattr(spec, f.name) for f in dc_fields(spec)}
            kwargs["dtype"] = torch.uint8
            return TQ4FullAttentionSpec(**kwargs)
        return spec

    Attention.get_kv_cache_spec = _tq4_get_kv_cache_spec
    logger.info("TQ4 FlashInfer backend registered as TQ4_FLASHINFER (packed cache)")
