from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

moto = pytest.importorskip("moto")
pytest.importorskip("s3fs")

# `s3fs` talks to S3 through `aiobotocore`'s async client. Moto's in-process
# `mock_aws` decorator only fabricates a *synchronous* botocore response
# object (plain `bytes` for `.content`), which aiobotocore's async endpoint
# then tries to `await` -- a structural incompatibility tracked upstream at
# getmoto/moto#8694, not something a version pin resolves. `ThreadedMotoServer`
# sidesteps this by serving real (loopback-only) HTTP that aiobotocore talks
# to like any other S3 endpoint, so we use it here instead of `mock_aws`.
# This is why `moto[server]` (not just `moto[s3]`) is required by the `dev`
# extra: the decorator-only `[s3]` extra doesn't pull in the Flask app the
# threaded server needs.


@pytest.fixture()
def s3_bucket(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from moto.server import ThreadedMotoServer

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{port}"

    import boto3

    client = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint_url)
    client.create_bucket(Bucket="wl-test")

    # `AWS_ENDPOINT_URL` is a standard botocore/aiobotocore config knob (env
    # var form of `endpoint_url`), so both our direct `boto3` setup call and
    # `SafetensorsWeightSource`'s internal fsspec/s3fs client transparently
    # point at the local moto server -- no production code needs to know
    # about test endpoints.
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint_url)
    try:
        yield "wl-test"
    finally:
        server.stop()


def test_s3_matches_local_bitexact(tmp_path: Path, s3_bucket: str) -> None:
    import s3fs

    from tests.fixtures_safetensors import write_single
    from weightlens.sources.safetensors import SafetensorsWeightSource

    tensors: dict[str, np.ndarray] = {
        "model.w": np.random.randn(16, 16).astype(np.float32)
    }
    local = str(tmp_path / "model.safetensors")
    write_single(local, tensors)

    fs = s3fs.S3FileSystem()
    fs.put(local, f"{s3_bucket}/model.safetensors")

    got = {
        lt.name: lt.values
        for lt in SafetensorsWeightSource(
            f"s3://{s3_bucket}/model.safetensors"
        ).iter_layers()
    }
    for name, arr in tensors.items():
        assert np.array_equal(got[name], arr)
