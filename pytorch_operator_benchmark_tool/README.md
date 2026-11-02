# CUDA operator benchmark

This project compares eager and `torch.compile` CUDA execution for matmul, softmax, LayerNorm, linear followed by GELU, and a pointwise-fusion workload. Inputs live in `config.json`, the benchmark entry point is `benchmark.py`, the operations live in `task.py`, and reusable timing, reporting, and plotting logic is in `helpers.py`.

## Run

From this directory, activate the environment containing the local PyTorch build and Triton, then run:

```bash
conda activate pytorch
PIP_EXTRA_INDEX_URL= PIP_CONFIG_FILE=/dev/null python -m pip install -r requirements.txt
python benchmark.py
```

For a named result directory:

```bash
python benchmark.py --output-dir benchmark_results/operators_run_1
```

To use another input file:

```bash
python benchmark.py --config config.json
```

## Files

- `config.json`: per-operation shapes, dtypes, timer iterations, samples, and `torch.compile` settings. Matmul and linear-GELU shapes are `M,K,N`; softmax, LayerNorm, and pointwise fusion shapes are `M,N`.
- `benchmark.py`: CLI parsing and benchmark orchestration.
- `task.py`: the operation being benchmarked.
- `helpers.py`: configuration validation, input generation, timing, result summaries, report generation, and log-scale SVG plotting.
- `COMPILE_PERFORMANCE_GUIDE.md`: eager-versus-compiled configuration and per-operator performance guidance.

Each run writes `environment.json`, `raw_results.csv`, `summary.json`, `report.pdf`, and one plot set per operation in `operator_plots/`. The summary and PDF include mean, median, Q1, Q3, p95, minimum, maximum, standard-deviation latency, compiled speedup, and CUDA memory measurements for every operation, input shape, dtype, and mode.
