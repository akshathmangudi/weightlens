# Changelog

## 0.3.0

- Analyze local BF16 safetensors values.
- Report inspection coverage and name every skipped eligible tensor. Global statistics use inspected tensors only.
- Preserve relative precision for quantile estimates across small and large weight values.
- Reject invalid safetensors layouts and restrict DCP metadata types and shard paths.
- Report an empty tensor as a warning. If the checkpoint has no tensor values, exit with code `1`.
- Return code `0` for completed reports, including `PARTIAL` reports and reports with findings. Invalid diagnostic cutoffs return code `2`.
- Disable the raw L2 abnormal-norm rule by default. The `--norm-threshold` flag enables the experimental rule.

DCP support remains experimental. Diagnostic cutoffs are starting values, not guarantees about model health.
