from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from safetensors.numpy import load_file

from tests.fixtures_safetensors import write_single
from weightlens.formats.safetensors_header import TensorSlice, decode_float_tensor
from weightlens.sources.safetensors import SafetensorsWeightSource

# ---------------------------------------------------------------------------
# Local helpers (this file must not depend on tests/fixtures_safetensors.py
# beyond write_single/write_sharded, and must not touch src/).
# ---------------------------------------------------------------------------


def _f32_bits(value: float) -> int:
    """Return the raw IEEE-754 bit pattern of a float32 as an unsigned int."""
    (bits,) = struct.unpack("<I", struct.pack("<f", value))
    return int(bits)


def _bf16_raw_from_floats(values: list[float]) -> bytes:
    """Truncate each float to bf16 (top 16 bits of its float32 encoding)."""
    out = b""
    for v in values:
        bits = _f32_bits(v)
        top16 = bits >> 16
        out += struct.pack("<H", top16)
    return out


def _bf16_expected_f32(values: list[float]) -> NDArray[np.float32]:
    """Hand-computed oracle: bf16-truncate then reinterpret as float32."""
    u16 = np.frombuffer(_bf16_raw_from_floats(values), dtype="<u2")
    widened = u16.astype(np.uint32) << 16
    return widened.view(np.float32).copy()


def _write_raw_safetensors(
    path: Path, entries: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    """Hand-construct a single-file safetensors blob for dtypes/shapes that
    the torch-based fixture helpers cannot express (e.g. BF16).

    ``entries`` maps tensor name -> (dtype_str, shape, raw_le_bytes).
    """
    header: dict[str, object] = {}
    body = b""
    for name, (dtype_str, shape, raw) in entries.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [len(body), len(body) + len(raw)],
        }
        body += raw
    header_json = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(header_json)) + header_json + body
    path.write_bytes(blob)


def _slice_for(
    name: str, dtype: str, shape: tuple[int, ...], begin: int, end: int
) -> TensorSlice:
    return TensorSlice(name=name, dtype=dtype, shape=shape, begin=begin, end=end)


# ---------------------------------------------------------------------------
# F16 / F32 / F64 bit-exact coverage vs the safetensors.numpy oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
def test_special_values_bit_exact_vs_oracle(
    tmp_path: Path, np_dtype: type[np.floating]
) -> None:
    """+/-inf, NaN, -0.0, and denormals must round-trip bit-exact."""
    raw_values = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        np.finfo(np_dtype).tiny,  # smallest positive normal
        np.finfo(np_dtype).max,  # largest finite magnitude
        -np.finfo(np_dtype).max,
    ]
    arr = np.array(raw_values, dtype=np_dtype)
    tensors: dict[str, np.ndarray] = {"w": arr}
    path = str(tmp_path / "special.safetensors")
    write_single(path, tensors)

    got = next(SafetensorsWeightSource(path).iter_layers())
    oracle = load_file(path)["w"]

    assert got.values.dtype == oracle.dtype
    got_bits = got.values.view(f"u{got.values.dtype.itemsize}")
    oracle_bits = oracle.view(f"u{oracle.dtype.itemsize}")
    assert np.array_equal(got_bits, oracle_bits), (
        f"{np_dtype}: bit patterns differ.\ngot={got.values}\noracle={oracle}"
    )
    # Sanity: NaN survives as NaN, -0.0 keeps its sign bit, inf keeps sign.
    assert np.isnan(got.values[6])
    assert math.copysign(1.0, float(got.values[3])) == -1.0
    assert math.isinf(float(got.values[4])) and got.values[4] > 0
    assert math.isinf(float(got.values[5])) and got.values[5] < 0


@pytest.mark.parametrize(
    "np_dtype,expected_str",
    [(np.float16, "float16"), (np.float32, "float32"), (np.float64, "float64")],
)
def test_denormals_survive_streaming(
    tmp_path: Path, np_dtype: type[np.floating], expected_str: str
) -> None:
    """Subnormal (denormal) magnitudes must not be flushed to zero."""
    tiny = np.finfo(np_dtype).tiny
    denorm = np.nextafter(np_dtype(0.0), np_dtype(1.0))  # smallest denormal
    assert 0.0 < float(denorm) < float(tiny)  # sanity: it really is subnormal

    tensors: dict[str, np.ndarray] = {
        "d": np.array([denorm, -denorm, tiny], dtype=np_dtype)
    }
    path = str(tmp_path / "denorm.safetensors")
    write_single(path, tensors)

    got = next(SafetensorsWeightSource(path).iter_layers())
    oracle = load_file(path)["d"]

    assert str(got.values.dtype) == expected_str
    assert np.array_equal(got.values, oracle)
    assert float(got.values[0]) != 0.0
    assert float(got.values[1]) != 0.0


# ---------------------------------------------------------------------------
# Shape edges: scalar (), 0-element, 3-D — across F16/F32/F64.
# ---------------------------------------------------------------------------


_ST_DTYPE_NAME = {np.float16: "F16", np.float32: "F32", np.float64: "F64"}


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
def test_scalar_shape_tensor(tmp_path: Path, np_dtype: type[np.floating]) -> None:
    # NB: tests.fixtures_safetensors.write_single cannot express a scalar
    # (shape ()) tensor: np.ascontiguousarray promotes 0-d arrays to shape
    # (1,) before torch.from_numpy ever sees them (documented numpy
    # behavior, not a weightlens bug). So we hand-build the raw file here
    # to genuinely exercise shape == () end to end.
    scalar = np.array(3.5, dtype=np_dtype)
    assert scalar.shape == ()
    raw = scalar.tobytes()
    path = tmp_path / "scalar.safetensors"
    _write_raw_safetensors(path, {"s": (_ST_DTYPE_NAME[np_dtype], (), raw)})

    got = next(SafetensorsWeightSource(str(path)).iter_layers())

    assert got.shape == ()
    assert got.values.shape == ()
    assert np.array_equal(got.values, scalar)
    assert float(got.values) == 3.5


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
def test_zero_element_tensor(tmp_path: Path, np_dtype: type[np.floating]) -> None:
    arr = np.zeros((0,), dtype=np_dtype)
    tensors: dict[str, np.ndarray] = {"empty": arr}
    path = str(tmp_path / "empty.safetensors")
    write_single(path, tensors)

    got = next(SafetensorsWeightSource(path).iter_layers())
    oracle = load_file(path)["empty"]

    assert got.shape == (0,)
    assert got.values.size == 0
    assert np.array_equal(got.values, oracle)


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
def test_zero_in_middle_dim_3d_tensor(
    tmp_path: Path, np_dtype: type[np.floating]
) -> None:
    """A 3-D tensor with a zero-sized middle dimension: still 0 elements."""
    arr = np.zeros((3, 0, 2), dtype=np_dtype)
    tensors: dict[str, np.ndarray] = {"z": arr}
    path = str(tmp_path / "zero3d.safetensors")
    write_single(path, tensors)

    got = next(SafetensorsWeightSource(path).iter_layers())
    oracle = load_file(path)["z"]

    assert got.shape == (3, 0, 2)
    assert got.values.size == 0
    assert np.array_equal(got.values, oracle)


@pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
def test_3d_tensor_bit_exact(tmp_path: Path, np_dtype: type[np.floating]) -> None:
    arr = np.linspace(-5.0, 5.0, 24, dtype=np.float64).astype(np_dtype).reshape(2, 3, 4)
    tensors: dict[str, np.ndarray] = {"cube": arr}
    path = str(tmp_path / "cube.safetensors")
    write_single(path, tensors)

    got = next(SafetensorsWeightSource(path).iter_layers())
    oracle = load_file(path)["cube"]

    assert got.shape == (2, 3, 4)
    assert np.array_equal(got.values, oracle)


# ---------------------------------------------------------------------------
# BF16: no numpy/safetensors oracle support, so we hand-build raw bytes from
# known floats and hand-compute the expected upcast-to-float32 arrays.
# ---------------------------------------------------------------------------


def test_bf16_special_values_bit_exact_hand_computed() -> None:
    values = [
        1.0,
        -1.0,
        0.5,
        -2.0,
        math.inf,
        -math.inf,
        math.nan,
        -0.0,
        0.0,
    ]
    raw = _bf16_raw_from_floats(values)
    sl = _slice_for("b", "BF16", (len(values),), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)
    expected = _bf16_expected_f32(values)

    assert decoded.dtype == np.float32
    decoded_bits = decoded.view(np.uint32)
    expected_bits = expected.view(np.uint32)
    assert np.array_equal(decoded_bits, expected_bits)

    assert math.isinf(float(decoded[4])) and decoded[4] > 0
    assert math.isinf(float(decoded[5])) and decoded[5] < 0
    assert np.isnan(decoded[6])
    assert math.copysign(1.0, float(decoded[7])) == -1.0
    assert math.copysign(1.0, float(decoded[8])) == 1.0


def test_bf16_denormal_bit_exact_hand_computed() -> None:
    """bf16 has its own (very coarse) denormal range: exponent field all
    zero, nonzero mantissa. Bit-construct one directly (no float literal
    truncates to this exactly) and confirm decode reproduces it exactly.
    """
    bf16_bits = 0x0001  # sign=0, exp=0000_0000, mantissa=0000001
    raw = struct.pack("<H", bf16_bits)
    sl = _slice_for("d", "BF16", (1,), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)

    expected_f32_bits = bf16_bits << 16
    (expected_val,) = struct.unpack("<f", struct.pack("<I", expected_f32_bits))

    assert decoded.dtype == np.float32
    assert decoded.view(np.uint32)[0] == expected_f32_bits
    assert float(decoded[0]) == expected_val
    assert 0.0 < float(decoded[0]) < 1e-38  # genuinely subnormal-ish/tiny


def test_bf16_large_magnitude_bit_exact_hand_computed() -> None:
    bf16_max_bits = 0x7F7F  # largest finite bf16 magnitude
    raw = struct.pack("<H", bf16_max_bits) + struct.pack(
        "<H", bf16_max_bits | 0x8000
    )
    sl = _slice_for("m", "BF16", (2,), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)

    expected_pos_bits = bf16_max_bits << 16
    expected_neg_bits = (bf16_max_bits | 0x8000) << 16
    assert decoded.view(np.uint32)[0] == expected_pos_bits
    assert decoded.view(np.uint32)[1] == expected_neg_bits
    assert float(decoded[0]) > 3.0e38
    assert float(decoded[1]) < -3.0e38


def test_bf16_upcasts_to_float32_dtype() -> None:
    raw = _bf16_raw_from_floats([1.0, 2.0, 3.0])
    sl = _slice_for("u", "BF16", (3,), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)
    assert decoded.dtype == np.dtype(np.float32)
    assert decoded.dtype != np.dtype(np.float16)


def test_bf16_scalar_shape() -> None:
    raw = _bf16_raw_from_floats([7.5])
    sl = _slice_for("s", "BF16", (), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)
    assert decoded.shape == ()
    assert decoded.dtype == np.float32
    expected = _bf16_expected_f32([7.5])[0]
    assert decoded.view(np.uint32) == expected.view(np.uint32)


def test_bf16_zero_element_shape() -> None:
    sl = _slice_for("e", "BF16", (0,), 0, 0)
    decoded = decode_float_tensor(b"", sl)
    assert decoded.shape == (0,)
    assert decoded.dtype == np.float32
    assert decoded.size == 0


def test_bf16_3d_tensor_bit_exact() -> None:
    values = [float(i) - 12.0 + 0.25 * i for i in range(24)]
    raw = _bf16_raw_from_floats(values)
    sl = _slice_for("cube", "BF16", (2, 3, 4), 0, len(raw))
    decoded = decode_float_tensor(raw, sl)
    expected = _bf16_expected_f32(values).reshape(2, 3, 4)

    assert decoded.shape == (2, 3, 4)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded.view(np.uint32), expected.view(np.uint32))


def test_bf16_end_to_end_streaming_bit_exact(tmp_path: Path) -> None:
    """Full pipeline: hand-built raw safetensors file -> SafetensorsWeightSource
    -> bit-exact vs hand-computed expected float32 array. Exercises the real
    header parsing + byte-range read path, not just decode_float_tensor.
    """
    values = [1.0, -2.0, 0.5, math.inf, -math.inf, math.nan, -0.0, 0.0]
    raw = _bf16_raw_from_floats(values)
    path = tmp_path / "bf16.safetensors"
    _write_raw_safetensors(
        path, {"w": ("BF16", (len(values),), raw)}
    )

    got = next(SafetensorsWeightSource(str(path)).iter_layers())
    expected = _bf16_expected_f32(values)

    assert got.dtype == "float32"
    assert got.values.dtype == np.float32
    got_bits = got.values.view(np.uint32)
    expected_bits = expected.view(np.uint32)
    # Compare non-NaN lanes bit-exact, then check NaN lane separately since
    # NaN != NaN would break array_equal.
    nan_mask = np.isnan(expected)
    assert np.array_equal(got_bits[~nan_mask], expected_bits[~nan_mask])
    assert np.isnan(got.values[nan_mask]).all()


def test_bf16_end_to_end_streaming_multiple_shapes(tmp_path: Path) -> None:
    """A single hand-built file with several BF16 tensors of different
    shapes (scalar, 1-D, 3-D, zero-element) streamed together.
    """
    scalar_vals = [42.0]
    vec_vals = [1.0, -1.0, 0.25]
    cube_vals = [float(i) * 0.5 for i in range(12)]

    entries: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "scalar": ("BF16", (), _bf16_raw_from_floats(scalar_vals)),
        "vec": ("BF16", (3,), _bf16_raw_from_floats(vec_vals)),
        "cube": ("BF16", (2, 2, 3), _bf16_raw_from_floats(cube_vals)),
        "empty": ("BF16", (0,), b""),
    }
    path = tmp_path / "multi_bf16.safetensors"
    _write_raw_safetensors(path, entries)

    source = SafetensorsWeightSource(str(path))
    got = {lt.name: lt.values for lt in source.iter_layers()}

    assert set(got) == {"scalar", "vec", "cube", "empty"}

    assert got["scalar"].shape == ()
    assert got["scalar"].view(np.uint32) == _bf16_expected_f32(scalar_vals)[0].view(
        np.uint32
    )

    assert got["vec"].shape == (3,)
    assert np.array_equal(
        got["vec"].view(np.uint32), _bf16_expected_f32(vec_vals).view(np.uint32)
    )

    assert got["cube"].shape == (2, 2, 3)
    expected_cube = _bf16_expected_f32(cube_vals).reshape(2, 2, 3)
    assert np.array_equal(
        got["cube"].view(np.uint32), expected_cube.view(np.uint32)
    )

    assert got["empty"].shape == (0,)
    assert got["empty"].size == 0
