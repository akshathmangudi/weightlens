from __future__ import annotations

import json


def parse_index(index_bytes: bytes) -> dict[str, str]:
    """Return ``{tensor_name: shard_filename}`` from a safetensors index.json."""
    doc = json.loads(index_bytes)
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index missing 'weight_map' object")
    return {str(k): str(v) for k, v in weight_map.items()}
