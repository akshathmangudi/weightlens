# PLAN: Configurable Diagnostic Thresholds + CI Hardening

## Execution order

Steps must run in this order — rules.py API change must land before cli.py wiring.

### 1. rules.py — Add threshold constructors

File: `src/weightlens/diagnostics/rules.py`

Add `__init__(self, threshold: float = <current>)` to each of the 4 rule classes. Store as `self._threshold`. Replace every hardcoded float literal in `check()` and in `message` f-strings with `self._threshold`.

Defaults match current values so no existing call sites break.

### 2. cli.py — Add CLI flags and wire through

File: `src/weightlens/cli.py`

Three sub-steps, must be done together:

a. `_build_parser()` — add `--variance-threshold`, `--spike-threshold`, `--norm-threshold`, `--sparsity-threshold` after the `--anon` flag. Use `type=float` with current defaults.

b. `_run_analyze()` — add 4 keyword-only params (`variance_threshold`, `spike_threshold`, `norm_threshold`, `sparsity_threshold`) with defaults. Pass them through to `_run_analyze_pytorch`, `_run_analyze_dcp`, `_run_analyze_safetensors`.

c. Each `_run_analyze_*` function — accept the 4 new params, pass them to rule constructors. Three functions to update: pytorch (line ~165), dcp (line ~232), safetensors (line ~286).

d. `run_cli()` — pass `args.variance_threshold` etc. to `_run_analyze()`.

### 3. test_performance_regression.py — Latency budget test

File: `tests/test_performance_regression.py`

Add `test_analyze_corrupted_checkpoint_completes_within_budget()`. Runs the full analyzer pipeline on `demo/checkpoints/corrupted_spike.pth` using `time.perf_counter()`. Asserts elapsed time < 3.0s.

Reuses existing imports (`time`, `Analyzer`, rule classes, etc. already in scope). Additional imports (`PyTorchCheckpointValidator`, `PyTorchWeightSource`, `BasicStatsEngine`, `StaticCheckpointValidator`) are imported inline at the test function to avoid polluting the module scope with unused symbols for other tests.

### 4. test.yml — CI matrix + remote endpoint job

File: `.github/workflows/test.yml`

a. Add `strategy.matrix.python-version: ["3.11", "3.12", "3.13"]` to `quick` job. Update job name to include Python version.

b. Rename `integration` job to `integration-moto`.

c. Add `integration-remote` job — runs `uv run pytest tests/test_remote_access.py -m integration -v` on every push and PR. Uses existing `test_real_s3_anon_byte_range` and `test_real_gcs_anon_byte_range` tests that hit public anonymous buckets.

### 5. README.md — One-liner

File: `README.md`

Add after the last Features bullet: `Conservative diagnostic thresholds to avoid false-positives on typical architectures; per-rule threshold configuration planned for v0.3`

### 6. GSD documents

Directory: `.planning/thresholds-config/`

Create `SPEC.md` (what we deliver, constraints, gotchas) and `PLAN.md` (this file — execution order, files touched, wiring instructions).

## Files changed

| File | Type | Risk |
|---|---|---|
| `src/weightlens/diagnostics/rules.py` | Code | Low — additive change, defaults preserved |
| `src/weightlens/cli.py` | Code | Low — new optional args, 3 wiring sites |
| `tests/test_performance_regression.py` | Test | Low — new test, isolated |
| `.github/workflows/test.yml` | CI | Medium — new job hits real endpoints, matrix multiplies runner usage |
| `README.md` | Docs | None |
| `.planning/thresholds-config/SPEC.md` | Docs | None |
| `.planning/thresholds-config/PLAN.md` | Docs | None |

## Files NOT changed

- `tests/` — zero existing test modifications (all no-arg constructors continue to work)
- `demos/demo.tape`, `demos/final.gif` — unchanged
- `src/weightlens/contracts/` — `DiagnosticRule` ABC unchanged
- `src/weightlens/reporters/rich_reporter.py` — `_build_summary` already done in prior round
- `script.py` — standalone debug script, left with no-arg constructors intentionally

## Verification

After all changes, run:

```bash
uv run ruff check src tests   # must pass (88-char line limit enforced)
uv run mypy                    # must pass (0 issues)
uv run pyright                 # must pass (0 errors)
uv run pytest                  # must pass (292 tests, 4 deselected)
uv run pytest tests/test_performance_regression.py -k "test_analyze_corrupted" -v  # latency test specifically
uv run lens analyze --help     # new flags appear in help output
```

## Gotchas for implementer

- Ruff E501 line length limit: CLI help strings must fit within 88 chars. Shorten "threshold" from help text to keep under limit — the `--help` output is unambiguous enough.
- The `replaceAll` edit flag in the editor tool will only replace exact string matches — if DCP and Safetensors constructor blocks have different indentation than PyTorch, they'll need separate edits. The PyTorch block at line ~188 uses 12-space indent; DCP at ~266 uses 16-space; Safetensors at ~312 uses 12-space. Verify each individually.
- `integration-remote` relies on anonymous S3/GCS reads. If AWS/GCP deprecate the public bucket URIs used in `test_remote_access.py`, this job silently breaks. No credential management needed — the tests use anonymous access by design.
- Python 3.11 requires `from __future__ import annotations` in every file — already the project convention. No additional compatibility work needed for the matrix expansion.
