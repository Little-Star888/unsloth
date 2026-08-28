# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for charging the tied-embedding output duplicate to the VRAM budget.

A model that ties its embeddings ships no ``output.weight``; llama.cpp
re-creates it from ``token_embd`` as TENSOR_DUPLICATED and a second vocabulary
matrix is really allocated. Sizing the load from the GGUF file alone therefore
UNDER-counts, which is the dangerous direction: it leaves the context search
believing there is VRAM the load will consume.

Anchored on measurement. gemma-4-E2B-it UD-Q4_K_XL sums to 3021.88 MiB of
tensors, and llama-server reported 3285.89 MiB of model buffers for it -- the
difference is 264.01 MiB against a ``token_embd`` of exactly 264.00 MiB, and the
two copies land on DIFFERENT devices (the original in CPU_Mapped, the duplicate
in CUDA0), which is why the duplicate is a VRAM cost and not merely a RAM one.

Pure: no GPU, no network, no subprocess. GGUFs are synthesised in a tmp_path.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# A minimal but real GGUF, so the probe is exercised through the same reader the
# product uses rather than a mock that could agree with a wrong implementation.
# ---------------------------------------------------------------------------

_GGUF_MAGIC = 0x46554747
_TYPE_F32 = 0


def _write_gguf(path: Path, tensors: "list[tuple[str, tuple[int, ...]]]") -> None:
    """Write a GGUF v3 with `tensors` as (name, shape), all f32, no KV pairs."""
    blobs: list[bytes] = []
    infos = bytearray()
    offset = 0
    for name, shape in tensors:
        raw = name.encode()
        infos += struct.pack("<Q", len(raw)) + raw
        infos += struct.pack("<I", len(shape))
        for dim in shape:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", _TYPE_F32)
        infos += struct.pack("<Q", offset)
        nbytes = 4
        for dim in shape:
            nbytes *= dim
        blobs.append(b"\0" * nbytes)
        offset += nbytes

    header = struct.pack("<II", _GGUF_MAGIC, 3) + struct.pack("<QQ", len(tensors), 1)
    # One KV pair: general.alignment, so the reader has a well-defined alignment
    # and the data section starts where the tensor offsets say it does.
    key = b"general.alignment"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", 32)

    body = header + kv + bytes(infos)
    pad = (-len(body)) % 32
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(b"\0" * pad)
        for blob in blobs:
            fh.write(blob)


@pytest.fixture(scope = "module")
def backend():
    pytest.importorskip("gguf")
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend


@pytest.fixture
def tied_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "tied.gguf"
    _write_gguf(
        path,
        [
            ("token_embd.weight", (8, 64)),  # 2048 bytes at f32
            ("blk.0.attn_q.weight", (8, 8)),
            ("blk.0.ffn_down.weight", (8, 8)),
        ],
    )
    return path


@pytest.fixture
def untied_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "untied.gguf"
    _write_gguf(
        path,
        [
            ("token_embd.weight", (8, 64)),
            ("output.weight", (8, 64)),
            ("blk.0.attn_q.weight", (8, 8)),
        ],
    )
    return path


def test_a_tied_model_is_charged_one_more_embedding_matrix(backend, tied_gguf):
    # 8 * 64 * 4 bytes. The duplicate is the WHOLE matrix, not a fraction of it.
    assert backend._tied_output_bytes(str(tied_gguf)) == 8 * 64 * 4


def test_a_model_shipping_its_own_output_is_charged_nothing(backend, untied_gguf):
    # The file already contains both tensors, so the file size covers the load
    # and adding anything would over-count. This is the Qwen3.6 / Qwen3.8 case.
    assert backend._tied_output_bytes(str(untied_gguf)) == 0


def test_the_charge_is_the_embedding_size_not_a_constant(backend, tmp_path):
    """Two tied models of different vocabulary sizes must differ.

    Guards against a fixed fudge factor, which would be right for one model and
    wrong for every other: the real spread across the shipped gemma quants is
    264 MiB (E2B UD-Q4_K_XL) to 924 MiB (31B UD-Q4_K_XL).
    """
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    _write_gguf(small, [("token_embd.weight", (8, 16))])
    _write_gguf(large, [("token_embd.weight", (8, 64))])
    assert backend._tied_output_bytes(str(large)) == 4 * backend._tied_output_bytes(str(small))


def test_a_split_gguf_is_read_across_every_shard(backend, tmp_path):
    """A shard cannot see its siblings, so the probe must open all of them.

    Reading only the named shard would call a model "tied" whenever its
    ``output.weight`` happens to live in a later one, and would miss a per-layer
    embedding that is not in shard 1 -- which is the Qwen3.8-Flash-Next case,
    where that tensor is 26.82 GiB and the whole point of the correction.
    """
    one = tmp_path / "m-00001-of-00002.gguf"
    two = tmp_path / "m-00002-of-00002.gguf"
    _write_gguf(one, [("token_embd.weight", (8, 64))])
    _write_gguf(two, [("output.weight", (8, 64)), ("blk.0.ffn_up.weight", (8, 8))])
    # output.weight is in shard 2, so this model is NOT tied.
    assert backend._tied_output_bytes(str(one)) == 0

    # Same shape without an output tensor anywhere: tied, and charged.
    three = tmp_path / "n-00001-of-00002.gguf"
    four = tmp_path / "n-00002-of-00002.gguf"
    _write_gguf(three, [("token_embd.weight", (8, 64))])
    _write_gguf(four, [("blk.0.ffn_up.weight", (8, 8))])
    assert backend._tied_output_bytes(str(three)) == 8 * 64 * 4


def test_the_per_layer_embedding_is_counted_from_a_later_shard(backend, tmp_path):
    """The largest host-pinned tensor is not required to be in shard 1."""
    one = tmp_path / "p-00001-of-00002.gguf"
    two = tmp_path / "p-00002-of-00002.gguf"
    _write_gguf(one, [("token_embd.weight", (8, 64))])
    _write_gguf(two, [("per_layer_token_embd.weight", (16, 64)), ("output.weight", (8, 64))])
    assert backend._host_pinned_weight_bytes(str(one)) == (8 * 64 * 4) + (16 * 64 * 4)


def test_host_pinned_covers_both_embedding_families(backend, tied_gguf, tmp_path):
    # token_embd alone on a model without per-layer embeddings.
    assert backend._host_pinned_weight_bytes(str(tied_gguf)) == 8 * 64 * 4

    ple = tmp_path / "ple.gguf"
    _write_gguf(
        ple,
        [
            ("token_embd.weight", (8, 64)),
            ("per_layer_token_embd.weight", (32, 64)),
            ("blk.0.ffn_up.weight", (8, 8)),
        ],
    )
    assert backend._host_pinned_weight_bytes(str(ple)) == (8 * 64 * 4) + (32 * 64 * 4)


def test_host_pinned_is_zero_for_an_unreadable_file(backend, tmp_path):
    junk = tmp_path / "junk2.gguf"
    junk.write_bytes(b"nope")
    assert backend._host_pinned_weight_bytes(str(junk)) == 0
    assert backend._host_pinned_weight_bytes(str(tmp_path / "gone.gguf")) == 0


def test_an_unreadable_file_costs_the_old_budget_rather_than_the_launch(backend, tmp_path):
    """The budget must never be the reason a load fails.

    A truncated or non-GGUF file returns 0, which is exactly the behaviour
    before this change, instead of propagating out of the context search.
    """
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    assert backend._tied_output_bytes(str(junk)) == 0
    assert backend._tied_output_bytes(str(tmp_path / "missing.gguf")) == 0


def test_the_probe_is_cached_on_file_identity_not_path(backend, tmp_path):
    """A model replaced in place must not serve the previous answer.

    The context search calls this once per candidate context, so it has to be
    cached; keying on the path alone would make a re-downloaded or re-quantised
    file keep its predecessor's charge.
    """
    path = tmp_path / "swapped.gguf"
    _write_gguf(path, [("token_embd.weight", (8, 64)), ("output.weight", (8, 64))])
    assert backend._tied_output_bytes(str(path)) == 0

    # Same name, different contents: now tied, and larger.
    _write_gguf(path, [("token_embd.weight", (8, 128))])
    assert backend._tied_output_bytes(str(path)) == 8 * 128 * 4


def test_the_budget_sizes_from_what_lands_in_vram(backend):
    """The call site adds the duplicate and subtracts the host-pinned bytes.

    Reads the source rather than driving a full load, which needs a GPU and a
    binary. What must hold is that `model_size` is neither the bare file size nor
    only half the correction.
    """
    import inspect

    src = inspect.getsource(backend)
    assert (
        "+ self._tied_output_bytes(model_path)" in src
    ), "the context budget no longer charges the tied-embedding duplicate"
    assert "- _host_pinned," in src, "the context budget no longer discounts host-pinned embeddings"
    # Unified memory must keep charging them: there the host pool IS the budget.
    assert "_amd_apu_wants_unified_memory(gpu_ids)" in src
    assert "_integrated_cuda_unified_memory(gpu_ids)" in src


# ---------------------------------------------------------------------------
# The one thing that outranks llama.cpp's unconditional host pinning.
# ---------------------------------------------------------------------------


def test_a_user_override_to_a_gpu_buffer_cancels_the_discount(backend):
    """`-ot` is applied before the fallback, so it really can move them.

    llama-model-loader.cpp:1182-1207 checks tensor_buft_overrides first. If a
    user has sent an input embedding to CUDA0 then those bytes ARE on the card
    and discounting them would over-commit, which is the only way this
    correction could hurt.
    """
    assert backend._override_moves_host_pinned(["-ot", "token_embd.weight=CUDA0"]) is True
    assert (
        backend._override_moves_host_pinned(["-ot", r"^per_layer_token_embd\.weight$=CUDA0"])
        is True
    )
    assert (
        backend._override_moves_host_pinned(["--override-tensor=token_embd.weight=CUDA0"]) is True
    )


def test_an_override_to_cpu_or_elsewhere_keeps_the_discount(backend):
    # Sending them to CPU is where llama.cpp puts them anyway.
    assert backend._override_moves_host_pinned(["-ot", "token_embd.weight=CPU"]) is False
    # Our own planner's patterns move FFN tensors, never the embeddings.
    assert (
        backend._override_moves_host_pinned(["-ot", r"^blk\.(1|2)\.ffn_down\.weight$=CPU"]) is False
    )
    assert backend._override_moves_host_pinned([]) is False
    assert backend._override_moves_host_pinned(None) is False
    # A bare flag with no value must not crash the budget.
    assert backend._override_moves_host_pinned(["-ot"]) is False
