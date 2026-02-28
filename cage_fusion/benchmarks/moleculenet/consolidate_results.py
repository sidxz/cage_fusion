#!/usr/bin/env python3
"""
Aggregate CAGE-Fusion MoleculeNet results across seeds.

- Scans an output root like: output/bert/Benchmark-v4
- For each dataset (subdirectory) and each seed (sub-subdirectory),
  reads test_metrics_<which_model>.json
- Extracts a chosen metric (e.g. 'test_auc' or 'test_pr')
- Computes mean and std across seeds
- Multiplies by a scale factor (default 100.0)
- Prints a LaTeX table to stdout with mean ± std

Example:
    python aggregate_molnet_results.py \
        --output-root output/bert/Benchmark-v4 \
        --metric test_auc \
        --which-model best_model \
        --decimals 1
"""

import os
import json
import argparse
import numpy as np


def find_metric_values(output_root, metric_key, which_model):
    """
    Walk the output_root and collect metric values per dataset.

    Expected layout:
      output_root/
        dataset_name/
          SEED_1/
            test_metrics_<which_model>.json
          SEED_2/
            test_metrics_<which_model>.json
          ...
    """
    results = {}  # dataset -> list of metric values

    for dataset in sorted(os.listdir(output_root)):
        dataset_dir = os.path.join(output_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        values = []
        for seed_dir in sorted(os.listdir(dataset_dir)):
            run_dir = os.path.join(dataset_dir, seed_dir)
            if not os.path.isdir(run_dir):
                continue

            metrics_path = os.path.join(
                run_dir, f"test_metrics_{which_model}.json"
            )
            if not os.path.exists(metrics_path):
                continue

            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                if metric_key not in data:
                    # allow per-task metrics if needed later
                    continue
                val = float(data[metric_key])
                values.append(val)
            except Exception as e:
                print(f"[WARN] Failed to read {metrics_path}: {e}")

        if values:
            results[dataset] = np.array(values, dtype=float)

    return results


def escape_latex(text: str) -> str:
    """Escape underscores for LaTeX dataset names."""
    return text.replace("_", r"\_")


def print_latex_table(results, metric_label, scale, decimals):
    """
    Print a LaTeX table with columns:
      Dataset | #Seeds | mean*scale ± std*scale
    """
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\small")
    print(
        rf"\caption{{Aggregated CAGE-Fusion performance across seeds "
        rf"({metric_label} $\times$ {int(scale)}, mean $\pm$ std).}}"
    )
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{\# Seeds} & \textbf{Score} \\")
    print(r"\midrule")

    for dataset, vals in sorted(results.items()):
        mean = vals.mean() * scale
        std = vals.std(ddof=0) * scale  # population std; ddof=1 if you prefer sample
        ds_name = escape_latex(dataset)
        print(
            rf"{ds_name} & {len(vals)} & "
            rf"{mean:.{decimals}f} $\pm$ {std:.{decimals}f} \\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:cagefusion_seed_agg}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate CAGE-Fusion MoleculeNet results across seeds."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root output directory (e.g., output/bert/Benchmark-v4)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_auc",
        help="Metric key in JSON (e.g., test_auc, test_pr, test_mcc).",
    )
    parser.add_argument(
        "--which-model",
        type=str,
        default="best_model",
        help="Which JSON to read: test_metrics_<which-model>.json "
             "(e.g., best_model or latest_model).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scaling factor applied to mean and std (default: 100.0).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Number of decimal places in the LaTeX table.",
    )
    args = parser.parse_args()

    results = find_metric_values(
        output_root=args.output_root,
        metric_key=args.metric,
        which_model=args.which_model,
    )

    if not results:
        print(
            f"[ERROR] No metrics found in {args.output_root} "
            f"for metric '{args.metric}' and which_model='{args.which_model}'."
        )
        return

    print_latex_table(
        results,
        metric_label=args.metric,
        scale=args.scale,
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
