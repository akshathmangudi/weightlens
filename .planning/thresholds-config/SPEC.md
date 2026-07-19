# SPEC: Configurable Diagnostic Thresholds + CI Hardening

## Deliverables

1. All 4 diagnostic rule thresholds accept configurable values via CLI flags, with defaults matching current hardcoded values — zero behavioral change for existing users
2. CI matrix covers Python 3.11 through 3.13 (was 3.12 only)
3. Real S3/GCS endpoint smoke test runs on every push and PR, gates CI on reachability
4. Latency budget test ensures analysis pipeline completes within 3.0s on the demo checkpoint, catching performance regressions
5. README acknowledges thresholds are conservative defaults; configuration planned for v0.3

## Non-goals

- No per-rule threshold customization in the API or config file — CLI-only for v0.3
- No threshold validation or auto-tuning against a checkpoint corpus — defaults remain the vetted values from first-principles analysis
- No changes to test files outside of the latency budget test — all existing tests use no-arg constructors and continue to pass

## Constraints

- Backward compatibility: all 15+ rule instantiation sites across src/ and tests/ must work unchanged with zero-arg constructors
- Line length: ruff E501 enforces 88-char limit on CLI help strings
- No new dependencies: the latency test reuses the existing performance regression file's imports
- CI job `integration-remote` must run on every push and PR, not just main — real endpoint reachability gates merge
- Latency budget of 3.0s provides 6x headroom above local runtime (~0.5s) to absorb CI runner variability

## Gotchas

- P² streaming quantile estimator produces approximate IQR — `iqr_layer_norm` on corrupted_spike.pth is ~111111.12, not the exact IQR of the 8 raw norms. This is expected streaming behavior, not a bug.
- `script.py` in root still uses no-arg rule constructors — intentionally left unchanged since it's a standalone debug script, not part of the package
- The `integration-remote` job hits live public S3/GCS buckets — if AWS/GCP change their public data URIs, this job will break. The test authors maintain the target URIs
- Threshold parameter names in CLI (`--variance-threshold`) differ slightly from rule class names (`ExplodingVarianceRule`) — the CLI names are user-facing, class names are internal
