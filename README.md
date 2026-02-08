# weightlens
Weightlens is an analysis tool for checkpoint weights. 

## What it solves
- Corruption detection (empty / partial failures, tensor access failures and NaN/zero floods) 
- Per-layer metrics (mean, std, min/max, L2 norm, sparsity and p99 absolute)
- Global distribution stats which are streamed to prevent OOM and memory crashes. 
- deterministic diagnostics for unhealthy layers. 

## What's next? 
- [ ] Improve diagnostics by bucketing components and softening constraints (bias, weights, norm_params, etc.)
- [ ] Integrate checkpoint diffing - compare regressions, drift, and training failures between two or more checkpoints
- [ ] Extend Weightlens for `h5`, `safetensors`, `joblib`, etc. 
- [ ] Research on deeper failure modes and detecting them accurately. 

## Performance 
| File size of `.pth` | Time taken | 
|---------------------|------------|
| ~ 200 MB | ~ 6 seconds | 
| ~ 1.5GB | ~ 16 seconds | 

## To use
Simply run `pip install weightlens` into your virtual environment and start by running `lens analyze <filename>.pth`

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

## Status
ALL TESTS AND LINT CHECKS PASS.

## Contributing
1. **Step 0**: Clone this repo. 
2. **Step 1**: Setup a virtual environment of your choice. The standard is uv as a `requirements.txt` does not exist here. 
3. **Step 2**: Run `uv pip install -e .[dev]` 
4. **Step 3**: Start contributing! 

If you would like to contribute, please do create Pull Requests. 

## Final Notes
This was a weekend project to work on, but it solves a real frustration by shedding some light onto how model 
checkpoints fail all the time. This library is NOT perfect. I will work on it!
