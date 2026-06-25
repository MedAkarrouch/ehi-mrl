#!/usr/bin/env python3
"""Plot Phase 3 FAISS IVF quality-efficiency sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = {
    "nq320k": {
        "label": "NQ320K",
        "sweep_arg": "nq_sweep",
        "exact_arg": "nq_exact",
        "default_sweep": "results/faiss_ivf/nq320k/sbert_distilbert_nli_stsb/sweep_summary.csv",
        "default_exact": "results/exact/nq320k/sbert_distilbert_nli_stsb/metrics_dev.json",
        "plots": ["Recall@100", "Recall@10", "MRR@10"],
        "primary_metric": "Recall@100",
        "prefix": "nq320k",
    },
    "scifact": {
        "label": "BEIR SciFact",
        "sweep_arg": "scifact_sweep",
        "exact_arg": "scifact_exact",
        "default_sweep": "results/faiss_ivf/beir_scifact/sbert_distilbert_nli_stsb/sweep_summary.csv",
        "default_exact": "results/exact/beir_scifact/sbert_distilbert_nli_stsb/metrics_test.json",
        "plots": ["nDCG@10", "Recall@100", "MRR@10"],
        "primary_metric": "nDCG@10",
        "prefix": "scifact",
    },
    "fiqa": {
        "label": "BEIR FiQA",
        "sweep_arg": "fiqa_sweep",
        "exact_arg": "fiqa_exact",
        "default_sweep": "results/faiss_ivf/beir_fiqa/sbert_distilbert_nli_stsb/sweep_summary.csv",
        "default_exact": "results/exact/beir_fiqa/sbert_distilbert_nli_stsb/metrics_test.json",
        "plots": ["nDCG@10", "Recall@100", "MRR@10"],
        "primary_metric": "nDCG@10",
        "prefix": "fiqa",
    },
}

METRICS = ["Hit@1", "MRR@10", "Recall@10", "Recall@100", "nDCG@10"]
BEST_BUDGETS = [1, 5, 10, 25, 50]
TERMINAL_BUDGETS = [5, 10, 25]
REQUIRED_CANONICAL_COLUMNS = [
    "dataset_name",
    "split",
    "nlist",
    "nprobe",
    "percent_docs_visited",
    "latency_ms_per_query",
    "Hit@1",
    "MRR@10",
    "Recall@10",
    "Recall@100",
    "nDCG@10",
]
COLUMN_ALIASES = {
    "%DocsVisited": "percent_docs_visited",
    "AvgDocsVisited": "avg_docs_visited",
    "LatencyMsPerQuery": "latency_ms_per_query",
}
METRIC_SLUGS = {
    "Recall@100": "recall100",
    "Recall@10": "recall10",
    "MRR@10": "mrr10",
    "nDCG@10": "ndcg10",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return repo_root() / target


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
        }
    )


def metric_slug(metric: str) -> str:
    return METRIC_SLUGS.get(metric, metric.lower().replace("@", "").replace(" ", "_"))


def load_exact_metric(path: Path, metric: str) -> float:
    if not path.is_file():
        raise RuntimeError(f"Missing exact metrics JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    metrics = data.get("metrics", data)
    if metric not in metrics:
        raise RuntimeError(f"Exact metrics file {path} is missing metric {metric}.")
    return float(metrics[metric])


def load_exact_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise RuntimeError(f"Missing exact metrics JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    metrics = data.get("metrics", data)
    return {metric: float(metrics[metric]) for metric in METRICS if metric in metrics}


def normalize_sweep_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = frame.rename(columns={source: target for source, target in COLUMN_ALIASES.items() if source in frame.columns})
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Sweep summary {path} is missing required column(s): {', '.join(missing)}")
    for column in ["nlist", "nprobe"]:
        frame[column] = frame[column].astype(int)
    numeric_columns = ["percent_docs_visited", "latency_ms_per_query", "avg_docs_visited", *METRICS]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["percent_docs_visited"].isna().any():
        raise RuntimeError(f"Sweep summary {path} has non-numeric percent_docs_visited values.")
    return frame.sort_values(["nlist", "percent_docs_visited", "nprobe"]).reset_index(drop=True)


def load_sweep(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing FAISS IVF sweep CSV: {path}")
    return normalize_sweep_columns(pd.read_csv(path), path)


def check_outputs(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"Refusing to overwrite existing plot output(s): {names}. Use --overwrite.")


def metric_axis_label(metric: str) -> str:
    return metric


def save_metric_plot(
    frame: pd.DataFrame,
    exact_value: float,
    dataset_label: str,
    output_prefix: str,
    metric: str,
    output_dir: Path,
    xscale: str,
    png_dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(layout="constrained")
    plot_frame = frame[frame["percent_docs_visited"] > 0].copy() if xscale == "log" else frame.copy()
    for nlist, group in plot_frame.groupby("nlist", sort=True):
        group = group.sort_values("percent_docs_visited")
        ax.plot(group["percent_docs_visited"], group[metric], marker="o", label=f"nlist={int(nlist)}")
        for _row_index, row in group.iterrows():
            ax.annotate(
                str(int(row["nprobe"])),
                (row["percent_docs_visited"], row[metric]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                alpha=0.75,
            )
    ax.axhline(exact_value, linestyle="--", color="black", linewidth=1.2, label="Phase 2 exact")
    ax.set_xscale(xscale)
    ax.set_xlabel("% documents visited")
    ax.set_ylabel(metric_axis_label(metric))
    ax.set_title(f"{dataset_label}: {metric} vs % documents visited")
    ax.set_ylim(bottom=0)
    ax.legend(title="IVF lists")
    ax.grid(True, alpha=0.25)
    base = output_dir / f"{output_prefix}_{metric_slug(metric)}_vs_docsvisited"
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return [svg_path, png_path]


def build_best_operating_points(
    sweeps: dict[str, pd.DataFrame],
    exact_metrics: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for key, spec in DATASETS.items():
        frame = sweeps[key]
        primary_metric = spec["primary_metric"]
        exact_value = exact_metrics[key].get(primary_metric, np.nan)
        for budget in BEST_BUDGETS:
            candidates = frame[frame["percent_docs_visited"] <= budget].copy()
            row = {
                "dataset_name": frame["dataset_name"].iloc[0] if len(frame) else spec["prefix"],
                "primary_metric": primary_metric,
                "budget_percent_docs_visited": budget,
                "selected_nlist": "",
                "selected_nprobe": "",
                "percent_docs_visited": "",
                "latency_ms_per_query": "",
                "Hit@1": "",
                "MRR@10": "",
                "Recall@10": "",
                "Recall@100": "",
                "nDCG@10": "",
                "exact_baseline_value": exact_value,
                "retention_vs_exact": "",
                "note": "",
            }
            if candidates.empty:
                row["note"] = "no sweep row under budget"
            else:
                candidates = candidates.sort_values([primary_metric, "percent_docs_visited"], ascending=[False, True])
                selected = candidates.iloc[0]
                selected_value = float(selected[primary_metric])
                row.update(
                    {
                        "selected_nlist": int(selected["nlist"]),
                        "selected_nprobe": int(selected["nprobe"]),
                        "percent_docs_visited": float(selected["percent_docs_visited"]),
                        "latency_ms_per_query": float(selected["latency_ms_per_query"]),
                        "Hit@1": float(selected["Hit@1"]),
                        "MRR@10": float(selected["MRR@10"]),
                        "Recall@10": float(selected["Recall@10"]),
                        "Recall@100": float(selected["Recall@100"]),
                        "nDCG@10": float(selected["nDCG@10"]),
                        "retention_vs_exact": selected_value / exact_value if exact_value else "",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def print_terminal_summary(best_points: pd.DataFrame, exact_metrics: dict[str, dict[str, float]]) -> None:
    for key, spec in DATASETS.items():
        primary_metric = spec["primary_metric"]
        exact_value = exact_metrics[key].get(primary_metric, np.nan)
        print("")
        print(spec["label"])
        print("Exact baseline:")
        print(f"  {primary_metric}: {exact_value:.6f}")
        dataset_rows = best_points[best_points["primary_metric"] == primary_metric]
        for budget in TERMINAL_BUDGETS:
            row = dataset_rows[
                (dataset_rows["dataset_name"].astype(str).str.contains(spec["prefix"], case=False, regex=False))
                & (dataset_rows["budget_percent_docs_visited"] == budget)
            ]
            if row.empty:
                row = dataset_rows[dataset_rows["budget_percent_docs_visited"] == budget].head(1)
            print(f"Best under {budget}% docs:")
            if row.empty or str(row.iloc[0]["note"]):
                note = row.iloc[0]["note"] if not row.empty else "not available"
                print(f"  {note}")
                continue
            selected = row.iloc[0]
            metric_value = float(selected[primary_metric])
            retention = float(selected["retention_vs_exact"]) if selected["retention_vs_exact"] != "" else np.nan
            print(
                "  "
                f"nlist={int(selected['selected_nlist'])}, "
                f"nprobe={int(selected['selected_nprobe'])}, "
                f"%DocsVisited={float(selected['percent_docs_visited']):.4f}, "
                f"latency={float(selected['latency_ms_per_query']):.4f} ms/query, "
                f"{primary_metric}={metric_value:.6f}, "
                f"retention={retention:.6f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nq-sweep", default=DATASETS["nq320k"]["default_sweep"])
    parser.add_argument("--scifact-sweep", default=DATASETS["scifact"]["default_sweep"])
    parser.add_argument("--fiqa-sweep", default=DATASETS["fiqa"]["default_sweep"])
    parser.add_argument("--nq-exact", default=DATASETS["nq320k"]["default_exact"])
    parser.add_argument("--scifact-exact", default=DATASETS["scifact"]["default_exact"])
    parser.add_argument("--fiqa-exact", default=DATASETS["fiqa"]["default_exact"])
    parser.add_argument("--output-dir", default="results/plots/faiss_ivf")
    parser.add_argument("--xscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweeps = {}
    exact_metrics = {}
    input_files = {}
    for key, spec in DATASETS.items():
        sweep_path = resolve_path(getattr(args, spec["sweep_arg"]))
        exact_path = resolve_path(getattr(args, spec["exact_arg"]))
        input_files[f"{key}_sweep"] = str(sweep_path)
        input_files[f"{key}_exact"] = str(exact_path)
        sweeps[key] = load_sweep(sweep_path)
        exact_metrics[key] = load_exact_metrics(exact_path)

    expected_outputs = []
    for key, spec in DATASETS.items():
        for metric in spec["plots"]:
            base = output_dir / f"{spec['prefix']}_{metric_slug(metric)}_vs_docsvisited"
            expected_outputs.extend([base.with_suffix(".svg"), base.with_suffix(".png")])
    expected_outputs.extend([output_dir / "best_operating_points.csv", output_dir / "plot_manifest.json"])
    check_outputs(expected_outputs, args.overwrite)

    output_files = []
    metrics_plotted = {}
    for key, spec in DATASETS.items():
        metrics_plotted[key] = spec["plots"]
        for metric in spec["plots"]:
            exact_value = load_exact_metric(resolve_path(getattr(args, spec["exact_arg"])), metric)
            output_files.extend(
                save_metric_plot(
                    sweeps[key],
                    exact_value,
                    spec["label"],
                    spec["prefix"],
                    metric,
                    output_dir,
                    args.xscale,
                    args.png_dpi,
                )
            )

    best_points = build_best_operating_points(sweeps, exact_metrics)
    best_path = output_dir / "best_operating_points.csv"
    best_points.to_csv(best_path, index=False)
    output_files.append(best_path)

    manifest = {
        "input_files": input_files,
        "output_files": [str(path) for path in [*output_files, manifest_path]],
        "generated_timestamp": pd.Timestamp.utcnow().isoformat(),
        "metrics_plotted": metrics_plotted,
        "xscale": args.xscale,
        "png_dpi": args.png_dpi,
    }
    manifest_path = output_dir / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output_files.append(manifest_path)
    print_terminal_summary(best_points, exact_metrics)
    print("")
    print(f"Wrote {len(output_files)} plot/summary files to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")
