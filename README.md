# weightlens
Weightlens is an analysis tool for checkpoint weights.

## What it solves
- Corruption detection (empty / partial failures, tensor access failures and NaN/zero floods)
- Per-layer metrics (mean, std, min/max, L2 norm, sparsity and p99 absolute)
- Global distribution stats, streamed to avoid OOM and memory crashes.
- Deterministic diagnostics for unhealthy layers.

## What's next?
- [x] Improve diagnostics by bucketing components and softening constraints (bias, weights, norm_params, etc.)
- [ ] Integrate checkpoint diffing: compare regressions, drift, and training failures between two or more checkpoints
- [ ] Extend Weightlens for `h5`, `safetensors`, `joblib`, etc. (DCP has been covered from a user request.)
- [ ] Research on deeper failure modes and detecting them accurately.

## Performance

Benchmarked on an ultrabook (Intel 4-core, 8GB LPDDR3, SATA SSD ~500 MB/s):

| Checkpoint | Format | Size | Tensors | Params | Wall time |
|---|---|---|---|---|---|
| BEiT-3 training checkpoint | `.pth` | 8 GB | 977 | 676M | ~29s |
| Mixtral MoE (multi-shard) | DCP | 70 GB | 456 | 20B | ~293s |

Performance is I/O-bound on SATA SSDs. On NVMe storage (3-7 GB/s), expect roughly proportional speedups. The `--num-workers` flag enables parallel stats computation which helps when I/O is not the bottleneck.

## To use
Simply run `pip install weightlens` into your virtual environment and start by running:

```bash
lens analyze <filename>.pth
lens analyze <dcp_directory> --format dcp
lens analyze <checkpoint>.pth --num-workers 2
```

## Remote checkpoints (safetensors)

Analyze checkpoints straight from object storage. Only tensor bytes are fetched, nothing is downloaded or consolidated:

```bash
pip install weightlens[s3]     # or [gcs], or [remote] for local fsspec only

lens analyze s3://bucket/model.safetensors
lens analyze s3://bucket/model.safetensors.index.json   # sharded
lens analyze gs://bucket/model.safetensors
```

Credentials use your existing AWS/GCS credential chain (env vars, `~/.aws/...`, instance roles). Weightlens stores no secrets. Remote `.pth` is supported via download-to-cache (`.pth` cannot be byte-ranged).

## Demo: corrupted checkpoints
Generate a clean checkpoint and two corrupted variants, then compare manual loading
versus Weightlens diagnostics.

```bash
python demo/make_clean_ckpt.py
python demo/corrupt_ckpt.py

lens analyze demo/checkpoints/clean.pth
lens analyze demo/checkpoints/corrupted_zero.pth
lens analyze demo/checkpoints/corrupted_spike.pth
```

If `lens` is not on your PATH, use `python -m weightlens.cli analyze ...` instead.

## Contributing
1. Clone this repo.
2. Set up a virtual environment. The standard is uv (no `requirements.txt`).
3. Run `uv pip install -e .[dev]`
4. Start contributing.

If you would like to contribute, please do create Pull Requests.

## Final Notes
This library is not perfect and is actively developed.
