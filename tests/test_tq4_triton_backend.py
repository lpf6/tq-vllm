"""Tests for TQ4 Triton backend.

These tests verify that the TQ4TritonBackend correctly implements
the TQ4 quantization for GPUs with compute capability 7.5+.
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if vLLM is not available
pytest.importorskip("vllm")

from turboquant_vllm.vllm.tq4_triton_backend import (
    TQ4TritonBackend,
    TQ4TritonImpl,
    TQ4FullAttentionSpec,
    _tq4_bytes_per_token,
    _tq4_bytes_per_token_kv,
)


class TestTQ4Constants:
    """Test TQ4 constant calculations."""

    def test_bytes_per_token(self):
        """Test byte calculation for different head sizes."""
        # For head_dim=128: 64 + 4 = 68 bytes
        assert _tq4_bytes_per_token(128) == 68
        # For head_dim=64: 32 + 4 = 36 bytes
        assert _tq4_bytes_per_token(64) == 36
        # For head_dim=256: 128 + 4 = 132 bytes
        assert _tq4_bytes_per_token(256) == 132

    def test_bytes_per_token_kv(self):
        """Test byte calculation for K+V combined."""
        # For head_dim=128: 68 * 2 = 136 bytes
        assert _tq4_bytes_per_token_kv(128) == 136
        # For head_dim=64: 36 * 2 = 72 bytes
        assert _tq4_bytes_per_token_kv(64) == 72


class TestTQ4FullAttentionSpec:
    """Test TQ4FullAttentionSpec."""

    def test_real_page_size_bytes(self):
        """Test page size calculation."""
        spec = TQ4FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
        )
        # 16 blocks * 8 heads * 136 bytes = 17408 bytes
        expected = 16 * 8 * 136
        assert spec.real_page_size_bytes == expected


class TestTQ4TritonBackend:
    """Test TQ4TritonBackend class."""

    def test_get_name(self):
        """Test backend name."""
        assert TQ4TritonBackend.get_name() == "TQ4_TRITON"

    def test_get_kv_cache_shape(self):
        """Test KV cache shape generation."""
        shape = TQ4TritonBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        # Expected: (100, 16, 136*8) = (100, 16, 1088)
        expected_bytes = 8 * 136
        assert shape == (100, 16, expected_bytes)

    def test_forward_includes_kv_cache_update(self):
        """Test that backend includes KV cache update."""
        assert TQ4TritonBackend.forward_includes_kv_cache_update is True

    def test_supports_mm_prefix(self):
        """Test MM prefix support."""
        assert TQ4TritonBackend.supports_mm_prefix() is True


class TestTQ4TritonImpl:
    """Test TQ4TritonImpl class."""

    @pytest.fixture
    def mock_layer(self):
        """Create a mock attention layer."""
        layer = nn.Module()
        layer._k_scale = torch.tensor([1.0])
        layer._v_scale = torch.tensor([1.0])
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        return layer

    def test_init(self):
        """Test initialization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        impl = TQ4TritonImpl(
            num_heads=32,
            head_size=128,
            scale=0.08838834764831843,  # 1/sqrt(128)
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        assert impl.num_heads == 32
        assert impl.head_size == 128
        assert impl.num_kv_heads == 8
        assert impl._half_D == 64
        assert impl._tq4_rotation.shape == (128, 128)
        assert impl._tq4_centroids.shape == (16,)
        assert impl._tq4_boundaries.shape == (15,)

    def test_compress_and_store(self):
        """Test compression and storage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        impl = TQ4TritonImpl(
            num_heads=32,
            head_size=128,
            scale=0.08838834764831843,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        # Create test data
        N, H, D = 4, 8, 128
        key = torch.randn(N, H, D, dtype=torch.float16, device="cuda")
        value = torch.randn(N, H, D, dtype=torch.float16, device="cuda")

        # Create cache
        num_blocks = 10
        block_size = 16
        total_bytes = impl._total_bytes
        kv_cache = torch.zeros(
            num_blocks, block_size, total_bytes, dtype=torch.uint8, device="cuda"
        )

        # Create slot mapping
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

        # Compress and store
        impl._compress_and_store(key, value, kv_cache, slot_mapping)

        # Verify cache was written (not all zeros)
        assert kv_cache.abs().sum() > 0

    def test_decompress_cache_paged(self):
        """Test decompression from paged cache."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        impl = TQ4TritonImpl(
            num_heads=32,
            head_size=128,
            scale=0.08838834764831843,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        # Initialize buffers
        num_blocks, block_size = 10, 16
        total_bytes = impl._total_bytes
        kv_cache = torch.zeros(
            num_blocks, block_size, total_bytes, dtype=torch.uint8, device="cuda"
        )

        # First, compress and store some data
        N, H, D = 16, 8, 128
        key = torch.randn(N, H, D, dtype=torch.float16, device="cuda")
        value = torch.randn(N, H, D, dtype=torch.float16, device="cuda")
        slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")
        impl._compress_and_store(key, value, kv_cache, slot_mapping)

        # Now decompress
        block_table = torch.tensor(
            [[0, 1, -1, -1]], dtype=torch.int32, device="cuda"
        )
        seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")

        out_k = torch.empty(64, H, D, dtype=torch.float16, device="cuda")
        out_v = torch.empty(64, H, D, dtype=torch.float16, device="cuda")

        key_cache, value_cache, remapped_bt = impl._decompress_cache_paged(
            kv_cache, block_table, seq_lens, torch.float16, out_k=out_k, out_v=out_v
        )

        # Verify shapes
        assert key_cache.shape[1:] == (block_size, H, D)
        assert value_cache.shape[1:] == (block_size, H, D)
        assert remapped_bt.shape == block_table.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
