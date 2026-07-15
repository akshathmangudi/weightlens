# weightlens

[![PyPI version](https://img.shields.io/pypi/v/weightlens.svg)](https://pypi.org/project/weightlens/)
[![CI](https://github.com/akshathmangudi/weightlens/actions/workflows/test.yml/badge.svg)](https://github.com/akshathmangudi/weightlens/actions)
[![Python versions](https://img.shields.io/pypi/pyversions/weightlens.svg)](https://pypi.org/project/weightlens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Analyze ML checkpoint weights without loading them into memory. Detect corruption in dead layers, extreme spikes, and NaN floods. Stream from S3/GCS via byte-range -- no download required.

## Quick start

```bash
pip install weightlens
lens analyze model.pth
```

```console
$ lens analyze demo/checkpoints/corrupted_spike.pth

Statistics for corrupted_spike.pth
================================================================================
file_size_bytes:    1,078,077
loadable:           true
is_empty:           false
tensor_count:       8
total_params:       268,650
corruption_flags:   none

Global Stats
================================================================================
mean:                    3.722
std:                     1929.327
p99:                     0.048
median_layer_variance:   0.000

Diagnostics (2)
================================================================================
   Severity   Rule            Layer           Message
   warn       exploding-var   conv2.weight    variance_ratio=37786784 >= 10.0
   error      extreme-spike   conv2.weight    spike_ratio=1291406248 >= 100.0
```

## Why weightlens?

`torch.load()` deserializes the entire checkpoint into memory before you can inspect a single tensor. Weightlens streams one tensor at a time through a chunked one-pass pipeline: statistics are computed via Welford's online algorithm on the fly, and histograms are accumulated in 1M-element chunks. Peak memory is bounded by chunk size, not checkpoint size.

For safetensors, weightlens memory-maps the file directly -- tensor data is never copied into a bytes buffer. The OS pages in exactly what numpy needs and no more. For PyTorch .pth files, weightlens uses the same chunked pipeline; memory is bounded by the deserialized checkpoint plus one chunk buffer.

Remote checkpoints get byte-range requests through S3/GCS credentials. Only tensor bytes are fetched -- the checkpoint is never downloaded. Format detection and format-specific streaming are handled automatically.

## Performance

Benchmarked on a MacBook M-series with NVMe SSD. All numbers measured with `/usr/bin/time -l` on real model checkpoints.

| Checkpoint | Format | Size | Tensors | Params | Time | Peak RSS |
|-----------|--------|------|---------|--------|------|-----------|
| ToyNet (demo) | .pth | 1 MB | 8 | 269K | 0.5s | 237 MB |
| SqueezeNet 1.1 | .pth | 5 MB | 52 | 1.2M | 0.6s | 237 MB |
| ResNet-18 | .pth | 45 MB | 102 | 11.7M | 0.8s | 324 MB |
| VGG-19 | .pth | 548 MB | 38 | 143.7M | 1.7s | 940 MB |
| Phi-2 | .index.json (sharded) | 5.6 GB | 453 | 2.8B | 28.6s | 659 MB |

Time is I/O-bound on local NVMe. The Phi-2 result is a cold read across 2 safetensors shards; the 659 MB peak RSS is constant for any safetensors file regardless of size. Remote first-run times include credential chain resolution. `--num-workers` parallelizes stats computation on larger models.

Historical data from the original README, measured on an Intel ultrabook with SATA SSD:

| Checkpoint | Format | Size | Tensors | Params | Time |
|-----------|--------|------|---------|--------|------|
| BEiT-3 training checkpoint | .pth | 8 GB | 977 | 676M | 29s |
| Mixtral MoE (multi-shard) | DCP | 70 GB | 456 | 20B | 293s |

## Features

- Detect dead layers (99.99%+ zeros), extreme spikes (100x above p99), exploding variance (10x above median), abnormal norms (5 sigma from IQR)
- Stream one tensor at a time through chunked one-pass statistics: Welford variance, incremental histogram, histogram-based p99
- Memory bounded by chunk size (1M elements ~= 2-8 MB), not file or tensor size
- Read safetensors from S3 or GCS via byte-range requests -- the checkpoint is never downloaded
- Identical results across .pth, .safetensors, and DCP formats
- First-run install: `pip install weightlens`

## Formats

| Format | Extension | Remote | Loading |
|--------|----------|--------|---------|
| PyTorch | .pth, .pt | Download to cache | Pickle deserialization + chunked stats |
| Safetensors | .safetensors | Byte-range | Memory-mapped views |
| Safetensors sharded | .index.json | Byte-range | Memory-mapped views per shard |
| DCP | directory | Offline | Byte-offset reads from shard files |

```bash
lens analyze model.pth
lens analyze model.safetensors
lens analyze model.safetensors.index.json
lens analyze checkpoint_dir --format dcp
```

Remote checkpoints use your existing AWS or GCS credentials. PyTorch CDN URLs work when the server supports Range headers:

```bash
pip install weightlens[remote]
lens analyze https://download.pytorch.org/models/resnet18-f37072fd.pth
lens analyze s3://bucket/model.safetensors
lens analyze s3://bucket/model.safetensors.index.json
lens analyze gs://bucket/model.safetensors
```

## Install

Python 3.11 or later.

```bash
pip install weightlens
pip install weightlens[s3]     # AWS S3
pip install weightlens[gcs]    # Google Cloud Storage
```

## Security

DCP metadata pickles are loaded through a whitelist-based unpickler blocking dangerous modules. PyTorch loading uses `weights_only=True` with no unsafe fallback; old-format .pth files that reject mmap fall back to non-mmap load (not unsafe unpickling). Path traversal is blocked in shard filenames for DCP and safetensors. Byte-range reads verify returned length. Remote downloads are capped at 50 GB.

## License

MIT
