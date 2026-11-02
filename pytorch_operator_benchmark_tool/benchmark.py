#!/usr/bin/env python3
import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

from helpers import benchmark_case, load_config, snapshot, summarize, write_csv, write_memory_svg, write_pdf_report, write_speedup_svg, write_statistic_plots, write_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eager and compiled CUDA operations.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if not torch.cuda.is_available():
        parser.error("CUDA is required; torch.cuda.is_available() is false")

    device = torch.device("cuda", 0)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tool_dir = Path(__file__).resolve().parent
    output_dir = arguments.output_dir or tool_dir / "benchmark_results" / f"operators_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = snapshot(device, config)
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}
    rows = []
    for operation, operation_config in config["operations"].items():
        for shape in operation_config["shapes"]:
            for dtype_name in config["dtypes"]:
                dtype = dtypes[dtype_name]
                if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    print(f"Skipping {operation} {shape} {dtype_name}: device does not support bf16", file=sys.stderr)
                    continue
                for mode in ("eager", "compiled"):
                    print(f"Running {operation} {shape} {dtype_name} {mode}", flush=True)
                    rows.extend(benchmark_case(operation, shape, dtype, mode, config, device))

    if not rows:
        raise RuntimeError("no benchmark cases ran")
    summary = summarize(rows)
    write_csv(rows, output_dir / "raw_results.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for operation in sorted({str(row["operation"]) for row in summary}):
        operation_summary = [row for row in summary if row["operation"] == operation]
        plot_dir = output_dir / "operator_plots" / operation
        plot_dir.mkdir(parents=True)
        write_svg(operation_summary, plot_dir / "mean_time.svg")
        write_speedup_svg(operation_summary, plot_dir / "compiled_speedup.svg")
        write_statistic_plots(operation_summary, plot_dir / "statistic_plots")
        write_memory_svg(operation_summary, plot_dir / "memory_usage.svg")
    write_pdf_report(summary, environment, output_dir / "report.pdf")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
