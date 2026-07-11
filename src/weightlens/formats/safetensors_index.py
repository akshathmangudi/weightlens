from __future__ import annotations

import json


def parse_index(index_bytes: bytes) -> dict[str, str]:
    """Return ``{tensor_name: shard_filename}`` from a safetensors index.json."""
    doc = json.loads(index_bytes)
    if not isinstance(doc, dict):
        raise ValueError("safetensors index is not a JSON object")
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index missing 'weight_map' object")
    result: dict[str, str] = {}
    for k, v in weight_map.items():
        v_str = str(v)
        if v_str.startswith("/") or ".." in v_str:
            raise ValueError(f"Potentially unsafe shard filename: {v_str!r}")
        result[str(k)] = v_str
    return result
