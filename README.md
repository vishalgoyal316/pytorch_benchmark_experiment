# pytorch_benchmark_experiment

Experiments comparing **eager** and **`torch.compile`** CUDA execution on common PyTorch operators (matmul, softmax, LayerNorm, linear+GELU, pointwise fusion).

- **Benchmark tool:** [`pytorch_operator_benchmark_tool/`](pytorch_operator_benchmark_tool/README.md) — run benchmarks, configs, reports, and sample results.
- **Analysis:** `benchmark_analysis.ipynb` and `benchmark_analysis-2.ipynb` at the repo root.

Requires CUDA and a PyTorch build with the compile backend you configure (see the tool README).
