# torch.compile Performance Guide

This project measures steady-state CUDA latency for the same preallocated random inputs in eager and compiled modes. It reports milliseconds per operation, memory usage, and compiled speedup:

```text
compiled speedup = eager mean latency / compiled mean latency
```

A speedup above `1.0x` means compiled execution is faster. A speedup below `1.0x` means eager execution is faster.

## Eager and compiled configurations

| Mode | Callable used | What is measured |
| --- | --- | --- |
| `eager` | Original task function | Normal PyTorch CUDA execution |
| `compiled` | `torch.compile(task, backend="inductor", mode="default")` | TorchDynamo/TorchInductor steady-state execution |

The compiled callable is created once per operator, shape, dtype, and mode. `torch.utils.benchmark.Timer` warms it before collecting the configured samples, so one-time graph compilation is not included in the reported latency. Tensor allocation and random-input generation also occur before timing.

## Current benchmark configuration

| Setting | Value |
| --- | --- |
| Backend | `inductor` |
| Compile mode | `default` |
| Iterations per sample | `100` |
| Samples per configuration | `50` |
| Dtypes | `fp32`, `bf16` |

`default` is the normal steady-state baseline. It is the right first configuration for comparing eager and compiled execution. `max-autotune` is a separate experiment for matrix-multiplication-heavy workloads: it can choose among more kernels, but increases compilation time and may use more resources.

## Operator behavior

| Operator | Inputs | Why compiled may be fast | Why compiled may be slow |
| --- | --- | --- | --- |
| Matmul | `left[M,K]`, `right[K,N]` | Possible template selection or fusion with neighboring operations in a larger graph | A standalone eager matmul already uses highly optimized cuBLAS/cuBLASLt kernels, leaving little work to improve |
| Softmax | `input[M,N]`, normalized over `N` | Generated kernels can improve some large, memory-bound shapes | Eager already has specialized softmax kernels; small shapes are dominated by compiler/dispatch overhead |
| LayerNorm | `input[M,N]`, normalized over `N` | Can fuse well with adjacent residual, bias, or activation operations in a larger graph | A standalone LayerNorm is already a specialized reduction kernel and has little fusion opportunity |
| Linear-GELU | `input[M,K]`, `weight[N,K]`, `bias[N]` | The bias and GELU may fuse into the linear epilogue, avoiding an intermediate output | The linear operation dominates runtime and eager GEMM is already optimized; benefit can be small |
| Pointwise fusion | `input[M,N]`, `bias[N]`, `residual[M,N]` | Add, multiply, add, SiLU, and clamp can become one generated kernel, eliminating intermediate reads, writes, and launches | Very small tensors may not recover the compile/dispatch overhead |

## Observed measurements

The existing GPU result at `benchmark_results/operators_20261016T080340Z/summary.json` contains matmul and softmax only. The strongest observed compiled improvement is large softmax:

| Operator | Shape | Dtype | Eager mean | Compiled mean | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Softmax | `8192x8192` | bf16 | 0.5380 ms | 0.3644 ms | 1.48x |
| Softmax | `8192x8192` | fp32 | 0.6125 ms | 0.4814 ms | 1.27x |

Standalone large matmul was approximately equal in eager and compiled modes in that run. This is expected for direct GEMM calls without surrounding operations to fuse.

## Interpreting the next full run

The current `config.json` profiles five operators: matmul, softmax, LayerNorm, linear-GELU, and pointwise fusion. Each gets its own plots and PDF pages.

Use the compiled-speedup plot first:

- Green bars above `1.0x`: compiled is faster.
- Red bars below `1.0x`: eager is faster.
- Compare only the same operator, shape, and dtype.
- Use the mean-latency and standard-deviation plots together. A small speedup close to the run-to-run variation is not a meaningful improvement.

If an expected fusion win does not appear, check for graph breaks and confirm that the task remains a single compiled region. Avoid data-dependent Python control flow and tensor-to-Python conversions such as `.item()` inside the task function.
