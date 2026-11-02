# Analysis of `operators_20261019T055450Z`

This analysis is based on [report.pdf](operators_20261019T055450Z/report.pdf), [summary.json](operators_20261019T055450Z/summary.json), and [environment.json](operators_20261019T055450Z/environment.json) from this run. It explains the observed steady-state mean-latency speedup:

```text
speedup = eager mean latency / compiled mean latency
```

Values above `1.0x` favor compiled mode. Values below `1.0x` favor eager mode.

## Run configuration

| Setting | Value |
| --- | --- |
| Backend | `inductor` |
| Compile mode | `default` |
| Iterations per sample | 100 |
| Trials per configuration | 50 |
| Dtypes | fp32, bf16 |
| Profiled operators | matmul, softmax, LayerNorm, linear-GELU, pointwise fusion |

Inputs are preallocated before timing. The benchmark measures steady-state execution, not random-input generation, allocation, or one-time compilation.

## Result summary

| Operator | Observed result | Interpretation |
| --- | --- | --- |
| Matmul | Large shapes are approximately `1.00x`; small shapes are slower compiled | Eager dispatches directly to an already optimized GEMM library kernel. There is no surrounding work for the compiler to fuse. |
| Softmax | Small shapes are slower compiled; `4096x4096` and `8192x8192` are faster compiled | At large shapes, the generated kernel amortizes dispatch overhead and improves the memory-bound workload. |
| LayerNorm | Compiled is slower for nearly all cases; one large fp32 case is approximately equal | This is an isolated, specialized reduction kernel with no adjacent operations to fuse. |
| Linear-GELU | Mixed; two fp32 sizes improve, most other cases are equal or slower | GEMM dominates the runtime. Epilogue fusion can help, but the benefit depends on shape and dtype. |
| Pointwise fusion | Compiled is strongly faster for every measured case | Inductor fuses the complete pointwise chain, removing intermediate tensors and several kernel launches. |

## Matmul

### Observed behavior

| Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| `512x512x512` | bf16 | 0.0105 ms | 0.0283 ms | 0.37x |
| `512x512x512` | fp32 | 0.0249 ms | 0.0340 ms | 0.73x |
| `2048x2048x2048` | bf16 | 0.0505 ms | 0.0507 ms | 1.00x |
| `4096x4096x4096` | fp32 | 6.9297 ms | 6.9299 ms | 1.00x |
| `6144x6144x6144` | bf16 | 1.1625 ms | 1.1625 ms | 1.00x |
| `6144x6144x6144` | fp32 | 22.9073 ms | 22.9080 ms | 1.00x |

### Why

This benchmark calls only `torch.matmul`. Eager mode already routes this directly to a highly tuned GEMM implementation. At large sizes, both modes spend almost all time inside that GEMM kernel, so their latencies are equal within measurement variation. At small sizes, the extra compiled call path is not amortized, so compiled is slower.

## Softmax

### Observed behavior

| Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| `512x512` | bf16 | 0.0065 ms | 0.0193 ms | 0.34x |
| `1024x1024` | fp32 | 0.0070 ms | 0.0250 ms | 0.28x |
| `2048x2048` | fp32 | 0.0219 ms | 0.0299 ms | 0.73x |
| `4096x4096` | bf16 | 0.1175 ms | 0.0987 ms | 1.19x |
| `4096x4096` | fp32 | 0.1177 ms | 0.1040 ms | 1.13x |
| `8192x8192` | bf16 | 0.5413 ms | 0.3643 ms | 1.49x |
| `8192x8192` | fp32 | 0.6147 ms | 0.4823 ms | 1.27x |

### Why

Softmax is an isolated reduction-and-normalization kernel. At small shapes, eager's specialized kernel is faster because compiled dispatch overhead is a large fraction of total time. At `4096x4096` and above, the work is large enough that overhead is amortized; the compiled kernel becomes faster. The best observed result is bf16 `8192x8192`, where compiled saves 0.1770 ms per call.

## LayerNorm

### Observed behavior

| Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| `512x512` | bf16 | 0.0106 ms | 0.0203 ms | 0.52x |
| `2048x2048` | fp32 | 0.0216 ms | 0.0348 ms | 0.62x |
| `4096x4096` | bf16 | 0.0518 ms | 0.1008 ms | 0.51x |
| `8192x8192` | bf16 | 0.1873 ms | 0.3535 ms | 0.53x |
| `8192x8192` | fp32 | 0.3822 ms | 0.3877 ms | 0.99x |

### Why

The standalone LayerNorm has no adjacent residual, bias, or activation operation in this benchmark. Therefore, compilation cannot gain much from fusion. Eager uses a specialized LayerNorm implementation, and its performance is better for all measured bf16 cases and most fp32 cases. The large fp32 case is nearly equal, not a meaningful compiled win.

## Linear-GELU

### Observed behavior

| Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| `512x512x512` | bf16 | 0.0185 ms | 0.0389 ms | 0.48x |
| `1024x1024x1024` | fp32 | 0.1647 ms | 0.1303 ms | 1.26x |
| `2048x2048x2048` | fp32 | 1.0773 ms | 1.0249 ms | 1.05x |
| `4096x4096x4096` | fp32 | 7.2431 ms | 7.7228 ms | 0.94x |
| `6144x6144x6144` | bf16 | 1.2016 ms | 1.2501 ms | 0.96x |
| `6144x6144x6144` | fp32 | 19.9117 ms | 25.2001 ms | 0.79x |

### Why

Linear-GELU consists of GEMM, bias addition, and GELU. Compiled mode can potentially fuse the pointwise bias/GELU epilogue, which explains the fp32 improvement at `1024x1024x1024` and smaller improvement at `2048x2048x2048`. However, GEMM dominates at larger shapes, and eager's matrix-multiplication path is already highly optimized. The pointwise fusion is then too small to offset the compiled path, especially for the largest fp32 case.

## Pointwise fusion

### Observed behavior

| Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| `2048x2048` | bf16 | 0.0581 ms | 0.0212 ms | 2.74x |
| `2048x2048` | fp32 | 0.0824 ms | 0.0243 ms | 3.40x |
| `4096x4096` | bf16 | 0.2348 ms | 0.0991 ms | 2.37x |
| `4096x4096` | fp32 | 0.3883 ms | 0.1097 ms | 3.54x |
| `8192x8192` | bf16 | 0.9186 ms | 0.3828 ms | 2.40x |
| `8192x8192` | fp32 | 1.5044 ms | 0.4258 ms | 3.53x |

### Why

The task computes:

```python
torch.clamp(F.silu((input + bias) * 0.5 + residual), -10, 10)
```

Eager mode typically launches separate kernels for the addition with bias, multiplication, addition with residual, SiLU, and clamp. Each step writes and later rereads a full intermediate tensor. Compiled mode fuses this chain into one generated kernel that reads `input`, `bias`, and `residual`, computes each element, and writes only the final value. The result is the strongest and most consistent speedup in this report: 2.37x to 3.54x.

## Practical conclusion

Use compiled mode for this pointwise-fusion workload and for large softmax shapes in this environment. Keep eager mode for standalone LayerNorm and treat standalone matmul as performance-equivalent at large sizes. Linear-GELU should be selected per shape and dtype: fp32 `1024x1024x1024` benefits, but the largest fp32 configuration does not.

These conclusions apply to this GPU, PyTorch build, backend, compile mode, and fixed input shapes. Re-run the benchmark after changing hardware, PyTorch, CUDA, shapes, or compile settings.
