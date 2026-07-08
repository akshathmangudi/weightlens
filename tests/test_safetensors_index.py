from __future__ import annotations

import json

import pytest

from weightlens.formats.safetensors_index import parse_index


def test_parse_index_returns_weight_map() -> None:
    doc = {
        "metadata": {"total_size": 100},
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        },
    }
    m = parse_index(json.dumps(doc).encode())
    assert m["a.weight"] == "model-00001-of-00002.safetensors"
    assert m["b.weight"] == "model-00002-of-00002.safetensors"


def test_parse_index_rejects_missing_weight_map() -> None:
    with pytest.raises(ValueError):
        parse_index(json.dumps({"metadata": {}}).encode())
