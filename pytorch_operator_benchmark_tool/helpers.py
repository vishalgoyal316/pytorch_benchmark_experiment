import argparse
import csv
import datetime
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import xml.sax.saxutils
from pathlib import Path

import torch
from torch.utils import benchmark

from task import TASK_DIMENSIONS, layer_norm, linear_gelu, matmul, parse_task_shape, pointwise_fusion, softmax


def load_config(path: Path) -> dict[str, object]:
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise argparse.ArgumentTypeError(f"config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("config must be a JSON object")
    required = {"operations", "dtypes", "iterations", "trials", "backend", "compile_mode"}
    missing = required.difference(config)
    if missing:
        raise argparse.ArgumentTypeError(f"config is missing: {', '.join(sorted(missing))}")
    if config["dtypes"] not in (["fp32"], ["bf16"], ["fp32", "bf16"], ["bf16", "fp32"]):
        raise argparse.ArgumentTypeError("dtypes must contain fp32 and/or bf16")
    if not isinstance(config["iterations"], int) or not isinstance(config["trials"], int):
        raise argparse.ArgumentTypeError("iterations and trials must be integers")
    if min(config["iterations"], config["trials"]) <= 0:
        raise argparse.ArgumentTypeError("iterations and trials must be positive")
    operations = config["operations"]
    if not isinstance(operations, dict) or not operations:
        raise argparse.ArgumentTypeError("operations must be a non-empty JSON object")
    for operation, operation_config in operations.items():
        if operation not in TASK_DIMENSIONS:
            raise argparse.ArgumentTypeError(f"unsupported operation {operation!r}")
        if not isinstance(operation_config, dict) or not isinstance(operation_config.get("shapes"), list):
            raise argparse.ArgumentTypeError(f"{operation} must contain a shapes list")
        if not operation_config["shapes"]:
            raise argparse.ArgumentTypeError(f"{operation} shapes must not be empty")
        for shape in operation_config["shapes"]:
            if not isinstance(shape, str):
                raise argparse.ArgumentTypeError(f"{operation} shapes must be strings")
            try:
                parse_task_shape(operation, shape)
            except ValueError as error:
                raise argparse.ArgumentTypeError(str(error)) from error
    return config


def nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,mig.mode.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def snapshot(device: torch.device, config: dict[str, object]) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_git_version": torch.version.git_version,
        "cuda_build_version": torch.version.cuda,
        "cuda_device": {
            "index": device.index,
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "nvidia_smi": nvidia_smi(),
        "environment": {
            name: os.environ[name]
            for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "TORCHINDUCTOR_CACHE_DIR")
            if name in os.environ
        },
        "benchmark": config,
    }


def time_operation(fn, inputs: tuple[torch.Tensor, ...], iterations: int, trials: int) -> list[float]:
    timer = benchmark.Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs": inputs})
    return [timer.timeit(number=iterations).mean * 1_000 for _ in range(trials)]


def capture_memory(fn, inputs: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, int]:
    output = fn(*inputs)
    torch.cuda.synchronize(device)
    del output
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    output = fn(*inputs)
    torch.cuda.synchronize(device)
    allocated_peak = torch.cuda.max_memory_allocated(device)
    reserved_peak = torch.cuda.max_memory_reserved(device)
    del output
    return {
        "peak_allocated_bytes": allocated_peak,
        "peak_reserved_bytes": reserved_peak,
        "incremental_peak_allocated_bytes": max(allocated_peak - allocated_before, 0),
        "incremental_peak_reserved_bytes": max(reserved_peak - reserved_before, 0),
    }


def benchmark_case(
    operation: str, shape: str, dtype: torch.dtype, mode: str, config: dict[str, object], device: torch.device
) -> list[dict[str, object]]:
    dimensions = parse_task_shape(operation, shape)
    if operation == "matmul":
        m, k, n = dimensions
        inputs = (
            torch.randn((m, k), device=device, dtype=dtype),
            torch.randn((k, n), device=device, dtype=dtype),
        )
        task = matmul
    elif operation == "softmax":
        inputs = (torch.randn(dimensions, device=device, dtype=dtype),)
        task = softmax
    elif operation == "layer_norm":
        inputs = (torch.randn(dimensions, device=device, dtype=dtype),)
        task = layer_norm
    elif operation == "linear_gelu":
        m, k, n = dimensions
        inputs = (
            torch.randn((m, k), device=device, dtype=dtype),
            torch.randn((n, k), device=device, dtype=dtype),
            torch.randn(n, device=device, dtype=dtype),
        )
        task = linear_gelu 
    elif operation == "pointwise_fusion":
        m, n = dimensions
        inputs = (
            torch.randn((m, n), device=device, dtype=dtype),
            torch.randn(n, device=device, dtype=dtype),
            torch.randn((m, n), device=device, dtype=dtype),
        )
        task = pointwise_fusion
    else:
        raise ValueError(f"unsupported operation {operation!r}")
    fn = task if mode == "eager" else torch.compile(task, backend=config["backend"], mode=config["compile_mode"])
    timings = time_operation(fn, inputs, config["iterations"], config["trials"])
    memory = capture_memory(fn, inputs, device)
    return [
        {
            "operation": operation,
            "shape": "x".join(str(dimension) for dimension in dimensions),
            "dtype": str(dtype).removeprefix("torch."),
            "mode": mode,
            "trial": trial,
            "time_ms": time_ms,
            **memory,
        }
        for trial, time_ms in enumerate(timings, start=1)
    ]


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["operation"]), str(row["shape"]), str(row["dtype"]), str(row["mode"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (operation, shape, dtype, mode), group_rows in sorted(groups.items()):
        samples = sorted(float(row["time_ms"]) for row in group_rows)
        if len(samples) == 1:
            q1 = q3 = p95 = samples[0]
        else:
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            p95 = statistics.quantiles(samples, n=100, method="inclusive")[94]
        summary.append(
            {
                "operation": operation,
                "shape": shape,
                "dtype": dtype,
                "mode": mode,
                "mean_ms": statistics.mean(samples),
                "median_ms": statistics.median(samples),
                "q1_ms": q1,
                "q3_ms": q3,
                "min_ms": samples[0],
                "max_ms": samples[-1],
                "p95_ms": p95,
                "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "trials": len(samples),
                **{
                    key: int(group_rows[0][key])
                    for key in (
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "incremental_peak_allocated_bytes",
                        "incremental_peak_reserved_bytes",
                    )
                    if key in group_rows[0]
                },
            }
        )
    return summary


def group_sort_key(group: tuple[tuple[str, str, str], object]) -> tuple[str, tuple[int, ...], str]:
    operation, shape, dtype = group[0]
    return operation, tuple(int(dimension) for dimension in shape.split("x")), dtype


def write_report(summary: list[dict[str, object]], environment: dict[str, object], output: Path) -> None:
    config = environment["benchmark"]
    device = environment["cuda_device"]
    results: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in summary:
        results.setdefault((str(row["operation"]), str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row

    lines = [
        "# PyTorch operator benchmark report",
        "",
        "## Run configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| PyTorch | {environment['pytorch']} |",
        f"| CUDA build | {environment['cuda_build_version']} |",
        f"| GPU | {device['name']} |",
        f"| Compute capability | {device['compute_capability']} |",
        f"| Dtypes | {', '.join(config['dtypes'])} |",
        *[f"| {operation} shapes | {', '.join(details['shapes'])} |" for operation, details in config["operations"].items()],
        f"| torch.compile backend | {config['backend']} |",
        f"| torch.compile mode | {config['compile_mode']} |",
        f"| Iterations per sample | {config['iterations']} |",
        f"| Samples per configuration | {config['trials']} |",
        "",
        "## Measurement",
        "",
        "Inputs are preallocated random CUDA tensors. Matmul uses `left[M, K]` and `right[K, N]`; softmax and LayerNorm use `input[M, N]` over the last dimension; linear-GELU uses `input[M, K]`, `weight[N, K]`, and `bias[N]`; pointwise fusion uses `input[M, N]`, `bias[N]`, and `residual[M, N]`. Each sample uses `torch.utils.benchmark.Timer.timeit`. The timer runs the operation for the configured number of iterations and synchronizes the accelerator as needed. Tensor allocation and compilation warmup are outside the reported measurements.",
        "",
        "Latency values are milliseconds per operation. Q1, Q3, and p95 use inclusive interpolation over the samples. Speedup is `eager_mean_ms / compiled_mean_ms`; values above 1 mean compiled was faster.",
        "",
        "## Latency statistics",
        "",
        "| Operation | Shape | Dtype | Mode | Mean | Median | Q1 | Q3 | p95 | Min | Max | Stddev |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['operation']} | {row['shape']} | {row['dtype']} | {row['mode']} | {float(row['mean_ms']):.4f} | "
            f"{float(row['median_ms']):.4f} | {float(row['q1_ms']):.4f} | {float(row['q3_ms']):.4f} | "
            f"{float(row['p95_ms']):.4f} | {float(row['min_ms']):.4f} | {float(row['max_ms']):.4f} | "
            f"{float(row['stddev_ms']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Eager and compiled comparison",
            "",
            "| Operation | Shape | Dtype | Eager mean (ms) | Compiled mean (ms) | Compiled speedup |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for (operation, shape, dtype), modes in sorted(results.items()):
        eager, compiled = modes.get("eager"), modes.get("compiled")
        if eager is None or compiled is None:
            continue
        eager_mean, compiled_mean = float(eager["mean_ms"]), float(compiled["mean_ms"])
        lines.append(f"| {operation} | {shape} | {dtype} | {eager_mean:.4f} | {compiled_mean:.4f} | {eager_mean / compiled_mean:.3f}x |")
    lines.extend(
        [
            "",
            "## Per-operation plots",
            "",
            "Each operation has separate mean-latency, compiled-speedup, memory, and per-statistic plots.",
            "",
        ]
    )
    statistic_plots = (
        ("Mean", "mean_ms"),
        ("Median", "median_ms"),
        ("Q1", "q1_ms"),
        ("Q3", "q3_ms"),
        ("P95", "p95_ms"),
        ("Minimum", "min_ms"),
        ("Maximum", "max_ms"),
        ("Standard deviation", "stddev_ms"),
    )
    for operation in sorted({str(row["operation"]) for row in summary}):
        plot_dir = Path("operator_plots") / operation
        lines.extend(
            [
                f"### {operation}",
                "",
                f"![Mean latency for {operation}]({plot_dir / 'mean_time.svg'})",
                "",
                f"![Compiled speedup for {operation}]({plot_dir / 'compiled_speedup.svg'})",
                "",
                f"[CUDA memory plot for {operation}]({plot_dir / 'memory_usage.svg'}).",
                "",
                "| Statistic | Plot |",
                "| --- | --- |",
                *[f"| {name} | [{name.lower()}]({plot_dir / 'statistic_plots' / f'{statistic}.svg'}) |" for name, statistic in statistic_plots],
                "",
            ]
        )
    lines.extend(
        [
            "[Download the complete PDF report](report.pdf).",
            "",
            "The raw trial measurements are in `raw_results.csv`; the machine and software snapshot is in `environment.json`; and the table source is in `summary.json`.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def write_svg(summary: list[dict[str, object]], output: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in summary:
        groups.setdefault((str(row["operation"]), str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row
    ordered_groups = sorted(groups.items(), key=group_sort_key)
    width, height, left, top, bottom = 1100, 700, 90, 70, 150
    chart_height = height - top - bottom
    minimum = min(max(float(row["mean_ms"]) - float(row["stddev_ms"]), 1e-12) for row in summary)
    maximum = max(float(row["mean_ms"]) + float(row["stddev_ms"]) for row in summary)
    minimum_power, maximum_power = math.floor(math.log10(minimum)), math.ceil(math.log10(maximum))
    if minimum_power == maximum_power:
        maximum_power += 1
    axis_minimum = 10**minimum_power
    scale = chart_height / (maximum_power - minimum_power)

    def y_position(value: float) -> float:
        return top + chart_height - (math.log10(max(value, axis_minimum)) - minimum_power) * scale

    bars = []
    labels = []
    group_width = (width - left - 40) / len(ordered_groups)
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    for group_index, ((operation, shape, dtype), modes) in enumerate(ordered_groups):
        center = left + group_width * (group_index + 0.5)
        for mode_index, mode in enumerate(("eager", "compiled")):
            row = modes.get(mode)
            if row is None:
                continue
            mean, deviation = float(row["mean_ms"]), float(row["stddev_ms"])
            x = center + (mode_index - 0.5) * 34 - 16
            y = y_position(mean)
            bar_height = top + chart_height - y
            error_top = y_position(mean + deviation)
            error_bottom = y_position(mean - deviation)
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="32" height="{bar_height:.1f}" fill="{colors[mode]}"/>')
            bars.append(f'<line x1="{x + 16:.1f}" y1="{error_top:.1f}" x2="{x + 16:.1f}" y2="{error_bottom:.1f}" stroke="#222"/>')
            bars.append(f'<line x1="{x + 9:.1f}" y1="{error_top:.1f}" x2="{x + 23:.1f}" y2="{error_top:.1f}" stroke="#222"/>')
            bars.append(f'<line x1="{x + 9:.1f}" y1="{error_bottom:.1f}" x2="{x + 23:.1f}" y2="{error_bottom:.1f}" stroke="#222"/>')
        labels.append(f'<text x="{center:.1f}" y="{height - 102}" text-anchor="middle">{xml.sax.saxutils.escape(operation)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 82}" text-anchor="middle">{xml.sax.saxutils.escape(shape)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 62}" text-anchor="middle">{xml.sax.saxutils.escape(dtype)}</text>')
    ticks = []
    for power in range(minimum_power, maximum_power + 1):
        value = 10**power
        y = y_position(value)
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:g}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="90" y="34" font-size="20" font-weight="bold">PyTorch operator benchmark</text>',
                '<text x="90" y="55">Log-scale mean time per operation; error bars show one standard deviation across trials.</text>',
                *ticks,
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
                f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 40}" y2="{top + chart_height}" stroke="#222"/>',
                *bars,
                *labels,
                '<rect x="810" y="25" width="14" height="14" fill="#4e79a7"/><text x="832" y="37">eager</text>',
                '<rect x="900" y="25" width="14" height="14" fill="#f28e2b"/><text x="922" y="37">compiled</text>',
                '<text x="20" y="365" transform="rotate(-90 20 365)" text-anchor="middle">time (ms, log scale)</text>',
                "</svg>",
            ]
        )
    )


def write_speedup_svg(summary: list[dict[str, object]], output: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in summary:
        groups.setdefault((str(row["operation"]), str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row
    speedups = []
    for group, modes in sorted(groups.items(), key=group_sort_key):
        eager, compiled = modes.get("eager"), modes.get("compiled")
        if eager is not None and compiled is not None:
            speedups.append((group, float(eager["mean_ms"]) / float(compiled["mean_ms"])))
    if not speedups:
        return
    width, height, left, top, bottom = 1100, 700, 90, 70, 150
    chart_height = height - top - bottom
    maximum = max(1.1, max(speedup for _, speedup in speedups) * 1.1)
    scale = chart_height / maximum
    baseline_y = top + chart_height - scale
    group_width = (width - left - 40) / len(speedups)
    bars = []
    labels = []
    for index, ((operation, shape, dtype), speedup) in enumerate(speedups):
        center = left + group_width * (index + 0.5)
        bar_height = speedup * scale
        y = top + chart_height - bar_height
        color = "#59a14f" if speedup >= 1 else "#e15759"
        bars.append(f'<rect x="{center - 18:.1f}" y="{y:.1f}" width="36" height="{bar_height:.1f}" fill="{color}"/>')
        bars.append(f'<text x="{center:.1f}" y="{y - 8:.1f}" text-anchor="middle">{speedup:.2f}x</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 102}" text-anchor="middle">{xml.sax.saxutils.escape(operation)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 82}" text-anchor="middle">{xml.sax.saxutils.escape(shape)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 62}" text-anchor="middle">{xml.sax.saxutils.escape(dtype)}</text>')
    ticks = []
    for tick in range(math.ceil(maximum) + 1):
        y = top + chart_height - tick * scale
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{tick:.0f}x</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="90" y="34" font-size="20" font-weight="bold">Compiled mean-latency speedup</text>',
                '<text x="90" y="55">Speedup = eager mean / compiled mean. Green is faster compiled execution; red is slower.</text>',
                *ticks,
                f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{width - 40}" y2="{baseline_y:.1f}" stroke="#222" stroke-dasharray="5 4"/>',
                *bars,
                *labels,
                '<text x="20" y="365" transform="rotate(-90 20 365)" text-anchor="middle">compiled speedup relative to eager</text>',
                "</svg>",
            ]
        )
    )


def write_distribution_svg(rows: list[dict[str, object]], summary: list[dict[str, object]], output: Path) -> None:
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    samples: dict[tuple[str, str, str], list[float]] = {}
    for row in summary:
        groups.setdefault((str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row
    for row in rows:
        key = (str(row["shape"]), str(row["dtype"]), str(row["mode"]))
        samples.setdefault(key, []).append(float(row["time_ms"]))
    width, height, left, top, bottom = 1100, 700, 90, 70, 130
    chart_height = height - top - bottom
    minimum = min(max(float(row["min_ms"]), 1e-12) for row in summary)
    maximum = max(float(row["max_ms"]) for row in summary)
    minimum_power, maximum_power = math.floor(math.log10(minimum)), math.ceil(math.log10(maximum))
    if minimum_power == maximum_power:
        maximum_power += 1
    axis_minimum = 10**minimum_power
    scale = chart_height / (maximum_power - minimum_power)

    def y_position(value: float) -> float:
        return top + chart_height - (math.log10(max(value, axis_minimum)) - minimum_power) * scale

    marks = []
    labels = []
    group_width = (width - left - 40) / len(groups)
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    for group_index, ((shape, dtype), modes) in enumerate(groups.items()):
        center = left + group_width * (group_index + 0.5)
        for mode_index, mode in enumerate(("eager", "compiled")):
            row = modes.get(mode)
            if row is None:
                continue
            x = center + (mode_index - 0.5) * 34 - 16
            center_x = x + 16
            minimum_y = y_position(float(row["min_ms"]))
            maximum_y = y_position(float(row["max_ms"]))
            q1_y = y_position(float(row["q1_ms"]))
            q3_y = y_position(float(row["q3_ms"]))
            median_y = y_position(float(row["median_ms"]))
            mean = float(row["mean_ms"])
            mean_y = y_position(mean)
            stddev = float(row["stddev_ms"])
            stddev_upper_y = y_position(mean + stddev)
            stddev_lower_y = y_position(mean - stddev)
            p95_y = y_position(float(row["p95_ms"]))
            marks.append(f'<line x1="{center_x:.1f}" y1="{maximum_y:.1f}" x2="{center_x:.1f}" y2="{minimum_y:.1f}" stroke="#222"/>')
            marks.append(f'<line x1="{x + 8:.1f}" y1="{maximum_y:.1f}" x2="{x + 24:.1f}" y2="{maximum_y:.1f}" stroke="#222"/>')
            marks.append(f'<line x1="{x + 8:.1f}" y1="{minimum_y:.1f}" x2="{x + 24:.1f}" y2="{minimum_y:.1f}" stroke="#222"/>')
            marks.append(f'<rect x="{x:.1f}" y="{q3_y:.1f}" width="32" height="{q1_y - q3_y:.1f}" fill="{colors[mode]}" fill-opacity="0.35" stroke="#222"/>')
            marks.append(f'<line x1="{center_x:.1f}" y1="{stddev_upper_y:.1f}" x2="{center_x:.1f}" y2="{stddev_lower_y:.1f}" stroke="#7b3294" stroke-dasharray="3 2"/>')
            marks.append(f'<line x1="{x:.1f}" y1="{median_y:.1f}" x2="{x + 32:.1f}" y2="{median_y:.1f}" stroke="#222" stroke-width="2"/>')
            sample_values = sorted(samples[(shape, dtype, mode)])
            offset = 22 / max(len(sample_values) - 1, 1)
            for sample_index, sample in enumerate(sample_values):
                sample_x = center_x + (sample_index - (len(sample_values) - 1) / 2) * offset
                marks.append(f'<circle cx="{sample_x:.1f}" cy="{y_position(sample):.1f}" r="2" fill="{colors[mode]}"/>')
            marks.append(f'<circle cx="{center_x:.1f}" cy="{mean_y:.1f}" r="4" fill="white" stroke="#7b3294" stroke-width="2"/>')
            marks.append(f'<polygon points="{center_x:.1f},{p95_y - 4:.1f} {center_x + 4:.1f},{p95_y:.1f} {center_x:.1f},{p95_y + 4:.1f} {center_x - 4:.1f},{p95_y:.1f}" fill="#d62728"/>')
        labels.append(f'<text x="{center:.1f}" y="{height - 82}" text-anchor="middle">{xml.sax.saxutils.escape(shape)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 62}" text-anchor="middle">{xml.sax.saxutils.escape(dtype)}</text>')
    ticks = []
    for power in range(minimum_power, maximum_power + 1):
        value = 10**power
        y = y_position(value)
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:g}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="90" y="34" font-size="20" font-weight="bold">PyTorch matmul latency statistics</text>',
                '<text x="90" y="55">Log scale; dots are trials, whiskers min/max, boxes Q1/Q3, black lines medians, white circles means, purple dashes mean +/- stddev, red diamonds p95.</text>',
                *ticks,
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
                f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 40}" y2="{top + chart_height}" stroke="#222"/>',
                *marks,
                *labels,
                '<rect x="745" y="25" width="14" height="14" fill="#4e79a7"/><text x="767" y="37">eager</text>',
                '<rect x="835" y="25" width="14" height="14" fill="#f28e2b"/><text x="857" y="37">compiled</text>',
                '<circle cx="955" cy="32" r="4" fill="white" stroke="#7b3294" stroke-width="2"/><text x="965" y="37">mean</text>',
                '<polygon points="1022,28 1026,32 1022,36 1018,32" fill="#d62728"/><text x="1031" y="37">p95</text>',
                '<text x="20" y="365" transform="rotate(-90 20 365)" text-anchor="middle">time (ms, log scale)</text>',
                "</svg>",
            ]
        )
    )


def write_single_config_svg(
    rows: list[dict[str, object]], summary: list[dict[str, object]], shape: str, dtype: str, output: Path
) -> None:
    display_shape = shape.replace(",", "x")
    result_dtype = {"fp32": "float32", "bf16": "bfloat16"}.get(dtype, dtype)
    selected_summary = [row for row in summary if row["shape"] == display_shape and row["dtype"] == result_dtype]
    selected_rows = [row for row in rows if row["shape"] == display_shape and row["dtype"] == result_dtype]
    if {row["mode"] for row in selected_summary} != {"eager", "compiled"}:
        raise RuntimeError(f"missing eager or compiled results for {shape} {dtype}")

    width, height, left, top, bottom = 1100, 380, 150, 80, 70
    chart_width = width - left - 40
    minimum = min(max(float(row["min_ms"]), 1e-12) for row in selected_summary)
    maximum = max(float(row["max_ms"]) for row in selected_summary)
    minimum_power, maximum_power = math.floor(math.log10(minimum)), math.ceil(math.log10(maximum))
    if minimum_power == maximum_power:
        maximum_power += 1
    axis_minimum = 10**minimum_power
    scale = chart_width / (maximum_power - minimum_power)

    def x_position(value: float) -> float:
        return left + (math.log10(max(value, axis_minimum)) - minimum_power) * scale

    summary_by_mode = {str(row["mode"]): row for row in selected_summary}
    samples = {mode: [] for mode in ("eager", "compiled")}
    for row in selected_rows:
        samples[str(row["mode"])].append(float(row["time_ms"]))
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    marks = []
    labels = []
    for index, mode in enumerate(("eager", "compiled")):
        row = summary_by_mode[mode]
        y = top + 90 + index * 110
        minimum_x = x_position(float(row["min_ms"]))
        maximum_x = x_position(float(row["max_ms"]))
        q1_x = x_position(float(row["q1_ms"]))
        q3_x = x_position(float(row["q3_ms"]))
        median_x = x_position(float(row["median_ms"]))
        mean = float(row["mean_ms"])
        mean_x = x_position(mean)
        stddev = float(row["stddev_ms"])
        stddev_min_x = x_position(mean - stddev)
        stddev_max_x = x_position(mean + stddev)
        p95_x = x_position(float(row["p95_ms"]))
        marks.append(f'<line x1="{minimum_x:.1f}" y1="{y}" x2="{maximum_x:.1f}" y2="{y}" stroke="#222"/>')
        marks.append(f'<line x1="{minimum_x:.1f}" y1="{y - 8}" x2="{minimum_x:.1f}" y2="{y + 8}" stroke="#222"/>')
        marks.append(f'<line x1="{maximum_x:.1f}" y1="{y - 8}" x2="{maximum_x:.1f}" y2="{y + 8}" stroke="#222"/>')
        marks.append(f'<rect x="{q1_x:.1f}" y="{y - 12}" width="{q3_x - q1_x:.1f}" height="24" fill="{colors[mode]}" fill-opacity="0.35" stroke="#222"/>')
        marks.append(f'<line x1="{stddev_min_x:.1f}" y1="{y}" x2="{stddev_max_x:.1f}" y2="{y}" stroke="#7b3294" stroke-dasharray="4 3" stroke-width="2"/>')
        marks.append(f'<line x1="{median_x:.1f}" y1="{y - 16}" x2="{median_x:.1f}" y2="{y + 16}" stroke="#222" stroke-width="2"/>')
        sample_values = sorted(samples[mode])
        for sample_index, sample in enumerate(sample_values):
            sample_y = y - 34 + sample_index * 7
            marks.append(f'<circle cx="{x_position(sample):.1f}" cy="{sample_y:.1f}" r="2.5" fill="{colors[mode]}"/>')
        marks.append(f'<circle cx="{mean_x:.1f}" cy="{y:.1f}" r="5" fill="white" stroke="#7b3294" stroke-width="2"/>')
        marks.append(f'<polygon points="{p95_x:.1f},{y - 6:.1f} {p95_x + 6:.1f},{y:.1f} {p95_x:.1f},{y + 6:.1f} {p95_x - 6:.1f},{y:.1f}" fill="#d62728"/>')
        labels.append(f'<text x="{left - 20}" y="{y + 4}" text-anchor="end" fill="{colors[mode]}">{mode}</text>')
    ticks = []
    for power in range(minimum_power, maximum_power + 1):
        value = 10**power
        x = x_position(value)
        ticks.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#ddd"/>')
        ticks.append(f'<text x="{x:.1f}" y="{height - 45}" text-anchor="middle">{value:g}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="150" y="34" font-size="20" font-weight="bold">Matmul latency statistics: {xml.sax.saxutils.escape(display_shape)} {xml.sax.saxutils.escape(dtype)}</text>',
                '<text x="150" y="55">Dots are trials; whiskers min/max; boxes Q1/Q3; black ticks median; white circles mean; purple dashes mean +/- stddev; red diamonds p95.</text>',
                '<rect x="800" y="20" width="14" height="14" fill="#4e79a7"/><text x="822" y="32">eager</text>',
                '<rect x="890" y="20" width="14" height="14" fill="#f28e2b"/><text x="912" y="32">compiled</text>',
                *ticks,
                f'<line x1="{left}" y1="{height - bottom}" x2="{width - 40}" y2="{height - bottom}" stroke="#222"/>',
                *marks,
                *labels,
                '<text x="625" y="365" text-anchor="middle">time (ms, log scale)</text>',
                "</svg>",
            ]
        )
    )


def write_statistic_svg(
    summary: list[dict[str, object]], shape: str, dtype: str, statistic: str, title: str, output: Path
) -> None:
    display_shape = shape.replace(",", "x")
    result_dtype = {"fp32": "float32", "bf16": "bfloat16"}.get(dtype, dtype)
    results = {str(row["mode"]): row for row in summary if row["shape"] == display_shape and row["dtype"] == result_dtype}
    if set(results) != {"eager", "compiled"}:
        raise RuntimeError(f"missing eager or compiled results for {shape} {dtype}")

    width, height, left, top, bottom = 600, 420, 80, 70, 80
    chart_height = height - top - bottom
    values = {mode: float(results[mode][statistic]) for mode in ("eager", "compiled")}
    maximum = max(values.values())
    scale = chart_height / (maximum * 1.15) if maximum else 1
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    bars = []
    labels = []
    for index, mode in enumerate(("eager", "compiled")):
        x = 180 + index * 180
        bar_height = values[mode] * scale
        y = top + chart_height - bar_height
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="100" height="{bar_height:.1f}" fill="{colors[mode]}"/>')
        labels.append(f'<text x="{x + 50}" y="{height - 52}" text-anchor="middle">{mode}</text>')
        labels.append(f'<text x="{x + 50}" y="{y - 8:.1f}" text-anchor="middle">{values[mode]:.4g} ms</text>')
    ticks = []
    for tick in range(6):
        value = maximum * 1.15 * tick / 5
        y = top + chart_height - value * scale
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.4g}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="80" y="32" font-size="20" font-weight="bold">{title}</text>',
                f'<text x="80" y="53">Matmul: {xml.sax.saxutils.escape(display_shape)} {xml.sax.saxutils.escape(dtype)}</text>',
                *ticks,
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
                f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 30}" y2="{top + chart_height}" stroke="#222"/>',
                *bars,
                *labels,
                '<text x="25" y="205" transform="rotate(-90 25 205)" text-anchor="middle">latency (ms)</text>',
                "</svg>",
            ]
        )
    )


def write_all_statistic_svg(summary: list[dict[str, object]], statistic: str, title: str, output: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in summary:
        groups.setdefault((str(row["operation"]), str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row
    ordered_groups = sorted(groups.items(), key=group_sort_key)
    width, height, left, top, bottom = 1100, 700, 90, 70, 150
    chart_height = height - top - bottom
    values = [float(row[statistic]) for row in summary if float(row[statistic]) > 0]
    minimum = min(values) if values else 1e-12
    maximum = max(values) if values else 1
    minimum_power, maximum_power = math.floor(math.log10(minimum)), math.ceil(math.log10(maximum))
    if minimum_power == maximum_power:
        maximum_power += 1
    axis_minimum = 10**minimum_power
    scale = chart_height / (maximum_power - minimum_power)

    def y_position(value: float) -> float:
        return top + chart_height - (math.log10(max(value, axis_minimum)) - minimum_power) * scale

    bars = []
    labels = []
    group_width = (width - left - 40) / len(ordered_groups)
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    for group_index, ((operation, shape, dtype), modes) in enumerate(ordered_groups):
        center = left + group_width * (group_index + 0.5)
        for mode_index, mode in enumerate(("eager", "compiled")):
            row = modes.get(mode)
            if row is None:
                continue
            x = center + (mode_index - 0.5) * 34 - 16
            y = y_position(float(row[statistic]))
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="32" height="{top + chart_height - y:.1f}" fill="{colors[mode]}"/>')
        labels.append(f'<text x="{center:.1f}" y="{height - 102}" text-anchor="middle">{xml.sax.saxutils.escape(operation)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 82}" text-anchor="middle">{xml.sax.saxutils.escape(shape)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 62}" text-anchor="middle">{xml.sax.saxutils.escape(dtype)}</text>')
    ticks = []
    for power in range(minimum_power, maximum_power + 1):
        value = 10**power
        y = y_position(value)
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:g}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="90" y="34" font-size="20" font-weight="bold">{title}</text>',
                '<text x="90" y="55">Log-scale latency across all shapes and dtypes.</text>',
                *ticks,
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
                f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 40}" y2="{top + chart_height}" stroke="#222"/>',
                *bars,
                *labels,
                '<rect x="810" y="25" width="14" height="14" fill="#4e79a7"/><text x="832" y="37">eager</text>',
                '<rect x="900" y="25" width="14" height="14" fill="#f28e2b"/><text x="922" y="37">compiled</text>',
                '<text x="20" y="365" transform="rotate(-90 20 365)" text-anchor="middle">latency (ms, log scale)</text>',
                "</svg>",
            ]
        )
    )


def write_statistic_plots(summary: list[dict[str, object]], output_dir: Path) -> None:
    plots = {
        "mean_ms": "Mean latency",
        "median_ms": "Median latency",
        "q1_ms": "Q1 latency",
        "q3_ms": "Q3 latency",
        "p95_ms": "P95 latency",
        "min_ms": "Minimum latency",
        "max_ms": "Maximum latency",
        "stddev_ms": "Latency standard deviation",
    }
    output_dir.mkdir(exist_ok=True)
    for statistic, title in plots.items():
        write_all_statistic_svg(summary, statistic, title, output_dir / f"{statistic}.svg")


def write_memory_svg(summary: list[dict[str, object]], output: Path) -> None:
    if not summary or "incremental_peak_allocated_bytes" not in summary[0]:
        return
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in summary:
        groups.setdefault((str(row["operation"]), str(row["shape"]), str(row["dtype"])), {})[str(row["mode"])] = row
    ordered_groups = sorted(groups.items(), key=group_sort_key)
    width, height, left, top, bottom = 1100, 700, 90, 70, 150
    chart_height = height - top - bottom
    maximum_bytes = max(float(row["incremental_peak_allocated_bytes"]) for row in summary)
    scale = chart_height / maximum_bytes if maximum_bytes else 1
    bars = []
    labels = []
    group_width = (width - left - 40) / len(ordered_groups)
    colors = {"eager": "#4e79a7", "compiled": "#f28e2b"}
    for group_index, ((operation, shape, dtype), modes) in enumerate(ordered_groups):
        center = left + group_width * (group_index + 0.5)
        for mode_index, mode in enumerate(("eager", "compiled")):
            row = modes.get(mode)
            if row is None:
                continue
            value = float(row["incremental_peak_allocated_bytes"])
            x = center + (mode_index - 0.5) * 34 - 16
            y = top + chart_height - value * scale
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="32" height="{value * scale:.1f}" fill="{colors[mode]}"/>')
        labels.append(f'<text x="{center:.1f}" y="{height - 102}" text-anchor="middle">{xml.sax.saxutils.escape(operation)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 82}" text-anchor="middle">{xml.sax.saxutils.escape(shape)}</text>')
        labels.append(f'<text x="{center:.1f}" y="{height - 62}" text-anchor="middle">{xml.sax.saxutils.escape(dtype)}</text>')
    ticks = []
    maximum_mib = maximum_bytes / 2**20
    for tick in range(6):
        value_mib = maximum_mib * tick / 5
        y = top + chart_height - value_mib / maximum_mib * chart_height if maximum_mib else top + chart_height
        ticks.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#ddd"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value_mib:.1f}</text>')
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="Arial, sans-serif" font-size="13">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="90" y="34" font-size="20" font-weight="bold">Incremental peak CUDA memory</text>',
                '<text x="90" y="55">Peak memory allocated above the pre-operation baseline.</text>',
                *ticks,
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
                f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 40}" y2="{top + chart_height}" stroke="#222"/>',
                *bars,
                *labels,
                '<rect x="810" y="25" width="14" height="14" fill="#4e79a7"/><text x="832" y="37">eager</text>',
                '<rect x="900" y="25" width="14" height="14" fill="#f28e2b"/><text x="922" y="37">compiled</text>',
                '<text x="20" y="365" transform="rotate(-90 20 365)" text-anchor="middle">incremental peak allocated memory (MiB)</text>',
                "</svg>",
            ]
        )
    )


def write_pdf_report(summary: list[dict[str, object]], environment: dict[str, object], output: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from svglib.svglib import svg2rlg

    def scaled_svg(path: Path, max_width: float, max_height: float):
        drawing = svg2rlg(str(path))
        if drawing is None:
            raise RuntimeError(f"could not load SVG: {path}")
        scale = min(max_width / drawing.width, max_height / drawing.height)
        drawing.scale(scale, scale)
        drawing.width *= scale
        drawing.height *= scale
        return drawing

    def page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(document.pagesize[0] - document.rightMargin, 20, f"Page {document.page}")
        canvas.restoreState()

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(output),
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="PyTorch operator benchmark report",
    )
    config = environment["benchmark"]
    device = environment["cuda_device"]
    profiled: dict[str, dict[str, set[str]]] = {}
    for row in summary:
        operation = str(row["operation"])
        details = profiled.setdefault(operation, {"shapes": set(), "dtypes": set(), "modes": set()})
        details["shapes"].add(str(row["shape"]))
        details["dtypes"].add(str(row["dtype"]))
        details["modes"].add(str(row["mode"]))
    profiled_rows = [["Operator", "Measured shapes", "Dtypes", "Modes"]]
    for operation, details in sorted(profiled.items()):
        profiled_rows.append(
            [
                operation,
                ", ".join(sorted(details["shapes"], key=lambda shape: tuple(int(dimension) for dimension in shape.split("x")))),
                ", ".join(sorted(details["dtypes"])),
                ", ".join(sorted(details["modes"])),
            ]
        )
    elements = [
        Paragraph("PyTorch operator benchmark report", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Profiled operations", styles["Heading2"]),
    ]
    profiled_table = Table(profiled_rows, repeatRows=1, colWidths=[120, 330, 120, 120])
    profiled_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#345b7e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b0b0")),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.extend(
        [
            profiled_table,
            Spacer(1, 16),
        Paragraph("Run configuration", styles["Heading2"]),
        ]
    )
    config_rows = [
        ["PyTorch", str(environment["pytorch"])],
        ["CUDA build", str(environment["cuda_build_version"])],
        ["GPU", str(device["name"])],
        ["Compute capability", str(device["compute_capability"])],
        ["Dtypes", ", ".join(config["dtypes"])],
        ["torch.compile backend", str(config["backend"])],
        ["torch.compile mode", str(config["compile_mode"])],
        ["Iterations per sample", str(config["iterations"])],
        ["Samples per configuration", str(config["trials"])],
    ]
    config_table = Table(config_rows, colWidths=[150, 540])
    config_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8edf4")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b0b0")),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.extend([config_table, Spacer(1, 16), Paragraph("Latency summary (ms)", styles["Heading2"])])
    summary_rows = [["Operation", "Shape", "Dtype", "Mode", "Mean", "Median", "Q1", "Q3", "P95", "Min", "Max", "Stddev"]]
    for row in summary:
        summary_rows.append(
            [
                str(row["operation"]),
                str(row["shape"]),
                str(row["dtype"]),
                str(row["mode"]),
                *[f"{float(row[key]):.4g}" for key in ("mean_ms", "median_ms", "q1_ms", "q3_ms", "p95_ms", "min_ms", "max_ms", "stddev_ms")],
            ]
        )
    summary_table = Table(summary_rows, repeatRows=1, colWidths=[60, 70, 48, 52, 52, 52, 52, 52, 52, 52, 52, 52])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#345b7e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b0b0")),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
                ("ALIGN", (4, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(summary_table)
    if "peak_allocated_bytes" in summary[0]:
        elements.extend([Spacer(1, 16), Paragraph("CUDA memory summary (MiB)", styles["Heading2"])])
        memory_rows = [["Operation", "Shape", "Dtype", "Mode", "Peak allocated", "Peak reserved", "Incremental allocated", "Incremental reserved"]]
        for row in summary:
            memory_rows.append(
                [
                    str(row["operation"]),
                    str(row["shape"]),
                    str(row["dtype"]),
                    str(row["mode"]),
                    *[f"{int(row[key]) / 2**20:.3f}" for key in ("peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes")],
                ]
            )
        memory_table = Table(memory_rows, repeatRows=1, colWidths=[65, 80, 50, 50, 90, 90, 110, 110])
        memory_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#345b7e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b0b0")),
                    ("FONT", (0, 0), (-1, -1), "Helvetica", 7),
                    ("ALIGN", (4, 1), (-1, -1), "RIGHT"),
                ]
            )
        )
        elements.append(memory_table)
    elements.append(PageBreak())

    statistic_plots = (
        ("mean_ms", "Mean latency"),
        ("median_ms", "Median latency"),
        ("q1_ms", "Q1 latency"),
        ("q3_ms", "Q3 latency"),
        ("p95_ms", "P95 latency"),
        ("min_ms", "Minimum latency"),
        ("max_ms", "Maximum latency"),
        ("stddev_ms", "Latency standard deviation"),
    )
    operations = sorted({str(row["operation"]) for row in summary})
    for operation_index, operation in enumerate(operations):
        plot_dir = output.parent / "operator_plots" / operation
        elements.extend(
            [
                Paragraph(f"{operation}: mean latency", styles["Heading1"]),
                Spacer(1, 12),
                scaled_svg(plot_dir / "mean_time.svg", 700, 480),
                PageBreak(),
                Paragraph(f"{operation}: compiled speedup", styles["Heading1"]),
                Spacer(1, 12),
                scaled_svg(plot_dir / "compiled_speedup.svg", 700, 480),
            ]
        )
        for page_index, plots in enumerate((statistic_plots[:4], statistic_plots[4:]), start=1):
            elements.extend(
                [
                    PageBreak(),
                    Paragraph(f"{operation}: latency statistics ({page_index}/2)", styles["Heading1"]),
                    Spacer(1, 12),
                    Table(
                        [
                            [scaled_svg(plot_dir / "statistic_plots" / f"{statistic}.svg", 330, 210) for statistic, _ in plots[:2]],
                            [scaled_svg(plot_dir / "statistic_plots" / f"{statistic}.svg", 330, 210) for statistic, _ in plots[2:]],
                        ],
                        colWidths=[345, 345],
                    ),
                ]
            )
        memory_plot = plot_dir / "memory_usage.svg"
        if memory_plot.exists():
            elements.extend(
                [
                    PageBreak(),
                    Paragraph(f"{operation}: incremental peak CUDA memory", styles["Heading1"]),
                    Spacer(1, 12),
                    scaled_svg(memory_plot, 700, 480),
                ]
            )
        if operation_index < len(operations) - 1:
            elements.append(PageBreak())
    document.build(elements, onFirstPage=page_number, onLaterPages=page_number)
