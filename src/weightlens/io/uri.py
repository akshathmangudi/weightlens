from __future__ import annotations

_PROTO_TO_EXTRA = {"s3": "s3", "gs": "gcs", "gcs": "gcs"}


def split_protocol(uri: str) -> str | None:
    if "://" in uri:
        return uri.split("://", 1)[0]
    return None


def is_remote(uri: str) -> bool:
    proto = split_protocol(uri)
    return proto is not None and proto != "file"


def extra_for_uri(uri: str) -> str:
    proto = split_protocol(uri) or ""
    return _PROTO_TO_EXTRA.get(proto, "remote")


def join_uri(base_uri: str, name: str) -> str:
    if not base_uri:
        return name
    return base_uri.rstrip("/") + "/" + name.lstrip("/")


def parent_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[0] if "/" in uri else ""


def anon_storage_options(uri: str) -> dict[str, object]:
    """Return the fsspec storage option that requests anonymous access.

    Each backend spells it differently; local paths need nothing.
    """
    proto = split_protocol(uri)
    if proto == "s3":
        return {"anon": True}
    if proto in ("gs", "gcs"):
        return {"token": "anon"}
    return {}
