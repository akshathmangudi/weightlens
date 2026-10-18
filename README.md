# weightlens

[![PyPI version](https://img.shields.io/pypi/v/weightlens.svg)](https://pypi.org/project/weightlens/)
[![CI](https://github.com/akshathmangudi/weightlens/actions/workflows/test.yml/badge.svg)](https://github.com/akshathmangudi/weightlens/actions)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Analyze ML checkpoint weights one tensor at a time. Report inspection gaps, dead layers, extreme values, and statistical anomalies across PyTorch, safetensors, and DCP checkpoints.

## Quick start

```bash
pip install weightlens
lens analyze model.pth
```

```console
$ lens analyze artifacts/checkpoints/corrupted_spike.pth

Statistics for corrupted_spike.pth
================================================================================
file_size_bytes:    1,078,077
loadable:           true
is_empty:           false
tensor_count:       8
total_params:       268,650
corruption_flags:   none

Inspection: COMPLETE
Coverage: 8 of 8 eligible tensors analyzed; 0 excluded; 0 skipped.

Global Stats (analyzed tensors only)
================================================================================
mean:                    3.722
std:                     1929.327
p99:                     0.000117969
median_layer_variance:   9.45793e-07
median_layer_norm:       0.019

Diagnostic thresholds
================================================================================
--variance-threshold:  10.0
--spike-threshold:     100.0
--norm-threshold:      off (experimental)
--sparsity-threshold:  0.9999

Diagnostics (2)
================================================================================
   Severity   Rule            Layer           Message
   warn       exploding-var   conv2.weight    variance_ratio=3.78e6 >= 10.0
   error      extreme-spike   conv2.weight    spike_ratio=1.29e9 >= 100.0

FINDINGS: 1 error-level finding, 1 warning
```

The report shows the active cutoffs and the disabled norm rule. A completed analysis returns exit code 0, even when it reports findings. An unreadable checkpoint returns a nonzero code. An invalid cutoff returns exit code 2.

NO FINDINGS means that no active diagnostic rule crossed its cutoff. It does not establish model quality. The severity column describes each finding, not the command's exit status.

PARTIAL means that Weightlens skipped an eligible tensor. The report names each skipped tensor and gives the reason. Global statistics cover only analyzed tensors. Intentional exclusions, such as integer counters, appear in the coverage section without making the inspection partial. An empty tensor receives a warning-level `empty-tensor` finding and appears as skipped. If every tensor is empty, Weightlens stops the inspection with exit code `1` because it has no values to analyze.

## Why weightlens?

`torch.load()` deserializes the entire checkpoint into memory. You cannot inspect a single tensor without paying the full memory cost first.

Weightlens reads tensors one at a time. Local F16, F32, and F64 safetensors use memory-mapped views. Local BF16 tensors are decoded to F32 one tensor at a time. Statistics run through Welford's online algorithm in 1M-element chunks. Some formats allocate a complete tensor during loading or conversion.

Safetensors checkpoints are byte-ranged from S3/GCS directly. Only tensor bytes are fetched and the file is never downloaded. PyTorch checkpoints download to a local cache first, then stream through the same chunked pipeline. Format detection and streaming are automatic.

## Performance

Benchmarked on a MacBook M-series with NVMe SSD. All numbers measured with `/usr/bin/time -l` on real model checkpoints.

| Checkpoint | Format | Size | Tensors | Params | Time | Peak RSS |
|-----------|--------|------|---------|--------|------|-----------|
| ToyNet (demo) | .pth | 1 MB | 8 | 269K | 0.5s | 237 MB |
| SqueezeNet 1.1 | .pth | 5 MB | 52 | 1.2M | 0.6s | 237 MB |
| ResNet-18 | .pth | 45 MB | 102 | 11.7M | 0.8s | 324 MB |
| VGG-19 | .pth | 548 MB | 38 | 143.7M | 1.7s | 940 MB |
| Phi-2 | .index.json (sharded) | 5.6 GB | 453 | 2.8B | 28.6s | 659 MB |

Time is I/O-bound on local NVMe. Phi-2 was a cold read across 2 safetensors shards. Peak RSS remained constant at ~659 MB regardless of file size due to memory-mapped tensor views and chunked processing. Remote first-run times include credential chain resolution. Use `--num-workers` to parallelize stats computation on larger models.

## Features

- Detect dead layers (99.99%+ zeros), NaN floods, extreme spikes (100x above p99), and exploding variance (10x above median). An experimental abnormal-norm warning is opt-in.
- Compute Welford variance and merge signed logarithmic quantile counts across tensors. The p99 estimates preserve relative precision across small and large weight scales.
- Process statistics in chunks of 1M elements. Peak memory also depends on the source format and the largest tensor.
- Read safetensors from S3 or GCS via byte-range requests. No full download needed.
- Apply the same analysis rules to .pth, .safetensors, and DCP formats.
- Default diagnostic cutoffs are starting values. They are not calibrated model-health guarantees. Set them with `--variance-threshold`, `--spike-threshold`, and `--sparsity-threshold`. Pass `--norm-threshold N` to enable the experimental raw-L2 warning.

## Formats

| Format | Extension | Remote | Loading |
|--------|----------|--------|---------|
| PyTorch | .pth, .pt | Download to cache | Pickle deserialization + chunked stats |
| Safetensors | .safetensors | Byte-range | Memory-mapped views |
| Safetensors sharded | .index.json | Byte-range | Memory-mapped views per shard |
| DCP | directory | Offline | Byte-offset reads from shard files |

DCP support is experimental. Weightlens reports any DCP tensor that it cannot inspect as skipped and explains the reason.

```bash
lens analyze model.pth
lens analyze model.safetensors
lens analyze model.safetensors.index.json
lens analyze checkpoint_dir --format dcp
```

Remote checkpoints use your existing AWS or GCS credentials. PyTorch CDN URLs download to a local cache first. Safetensors URLs use byte-range reads when the server supports Range headers:

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

DCP metadata pickles use an allowlist of PyTorch metadata globals. Metadata that needs other globals fails to load. Only analyze DCP metadata from sources that you trust. PyTorch loading uses `weights_only=True` with no unsafe fallback. Old-format .pth files that reject mmap fall back to non-mmap load. DCP shard paths must stay inside the checkpoint directory. Safetensors shard paths block traversal. Byte-range reads make sure that the response length is correct. Remote downloads are capped at 50 GB.

## License

MIT
