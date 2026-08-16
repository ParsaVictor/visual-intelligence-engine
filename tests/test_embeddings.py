"""Regression tests for P2-5: raw float32 storage instead of pickle."""

from __future__ import annotations

import numpy as np
import pytest

from vie.embeddings import cosine_similarity, from_blob, l2_normalise, to_blob


def test_roundtrip_preserves_values() -> None:
    vector = np.random.default_rng(0).normal(size=512).astype(np.float32)
    assert np.allclose(from_blob(to_blob(vector)), vector, atol=1e-6)


def test_blob_size_is_four_bytes_per_dimension() -> None:
    """A 512-d ArcFace embedding is 2 KB. Pickling a Python list was several times that."""
    assert len(to_blob(np.zeros(512, dtype=np.float32))) == 512 * 4


def test_blob_contains_no_pickle_opcodes() -> None:
    """Loading a pickle from an untrusted index file is arbitrary code execution."""
    blob = to_blob(np.ones(64, dtype=np.float32))
    assert not blob.startswith(b"\x80")  # pickle protocol 2+ marker


def test_truncated_blob_is_rejected() -> None:
    with pytest.raises(ValueError, match="multiple"):
        from_blob(to_blob(np.ones(8, dtype=np.float32))[:-1])


def test_empty_blob_is_rejected() -> None:
    with pytest.raises(ValueError):
        from_blob(b"")


def test_empty_vector_is_rejected() -> None:
    with pytest.raises(ValueError):
        to_blob(np.array([], dtype=np.float32))


def test_nan_is_rejected() -> None:
    """A NaN embedding silently poisons every similarity it takes part in."""
    with pytest.raises(ValueError, match="NaN"):
        to_blob(np.array([1.0, np.nan, 2.0], dtype=np.float32))


def test_normalised_vector_has_unit_length() -> None:
    v = l2_normalise(np.array([3.0, 4.0], dtype=np.float32))
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)


def test_dot_product_equals_cosine_for_normalised_vectors() -> None:
    """This is why the pipeline can use a plain dot product as its metric."""
    rng = np.random.default_rng(1)
    a = l2_normalise(rng.normal(size=128).astype(np.float32))
    b = l2_normalise(rng.normal(size=128).astype(np.float32))
    assert float(np.dot(a, b)) == pytest.approx(cosine_similarity(a, b), abs=1e-6)


def test_identical_vectors_have_similarity_one() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_dimension_mismatch_is_rejected() -> None:
    """The original silently skipped mismatched rows instead of reporting them."""
    with pytest.raises(ValueError, match="mismatch"):
        cosine_similarity(np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32))


def test_zero_vector_normalisation_is_rejected() -> None:
    with pytest.raises(ValueError):
        l2_normalise(np.zeros(4, dtype=np.float32))
