"""Calculate fold-matched binary metrics for SMRTnet and Thesis A predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_THESIS_ROOT = Path(__file__).resolve().parents[2]
_THESIS_SRC = _THESIS_ROOT / "src"
if str(_THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(_THESIS_SRC))

from affinity_experiment.metrics import calculate_metrics


_METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


class ComparisonError(RuntimeError):
    """Prediction files do not describe a matched five-fold binary evaluation."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ComparisonError(f"could not read prediction file {path}: {exc}") from exc


def _write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _binary_probability(row: Mapping[str, str], *, source: str, key: str) -> tuple[int, float]:
    try:
        label = int(row["label"])
        probability = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"{source} prediction row has invalid label/probability") from exc
    if label not in (0, 1) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ComparisonError(f"{source} prediction row must contain a binary label and [0, 1] score")
    return label, probability


def _metric_payload(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    count = len(labels)
    return calculate_metrics(
        binding_targets=labels,
        binding_available=[True] * count,
        binding_probabilities=probabilities,
        pkd_targets=[0.0] * count,
        pkd_available=[False] * count,
        pkd_predictions=[0.0] * count,
        threshold=0.5,
    )["classification"]


def _summary(folds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_name in ("smrtnet", "thesis_a"):
        summary[model_name] = {}
        for metric in _METRICS:
            values = [
                float(fold[model_name][metric])
                for fold in folds.values()
                if fold[model_name][metric] is not None
                and math.isfinite(float(fold[model_name][metric]))
            ]
            summary[model_name][metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.pstdev(values) if values else None,
                "defined_fold_count": len(values),
            }
    return summary


def compare_predictions(
    smrtnet_run_dir: str | Path,
    thesis_predictions: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Compare pair scores on the per-fold intersection of held-out eligible example IDs."""

    run_dir = Path(smrtnet_run_dir).expanduser().resolve()
    baseline_path = Path(thesis_predictions).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "comparison"
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"could not read SMRTnet run summary under {run_dir}: {exc}") from exc
    if summary.get("fold_ids_completed") != [0, 1, 2, 3, 4] or summary.get("smoke"):
        raise ComparisonError("comparison requires all five non-smoke held-out fold runs")
    strategy = summary.get("strategy")
    baseline_by_fold: dict[int, dict[str, tuple[int, float, dict[str, str]]]] = {
        fold_id: {} for fold_id in range(5)
    }
    for row in _read_csv(baseline_path):
        if row.get("partition") != "test" or row.get("label_available", "true").lower() not in {
            "1", "true", "yes"
        }:
            continue
        try:
            fold_id = int(row["fold_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ComparisonError("Thesis A prediction row is missing fold_id") from exc
        if fold_id not in baseline_by_fold:
            continue
        example_id = row.get("example_id", "")
        if not example_id:
            raise ComparisonError("Thesis A prediction row is missing example_id")
        label, probability = _binary_probability(
            row, source="Thesis A", key="binding_probability"
        )
        if example_id in baseline_by_fold[fold_id]:
            raise ComparisonError(f"duplicate Thesis A test prediction for fold={fold_id}, {example_id}")
        baseline_by_fold[fold_id][example_id] = (label, probability, row)

    fold_results: dict[str, dict[str, Any]] = {}
    matched_predictions: list[dict[str, Any]] = []
    for fold_id in range(5):
        pair_path = run_dir / ("fold_" + str(fold_id)) / "pair_predictions.csv"
        model_by_id: dict[str, tuple[int, float, dict[str, str]]] = {}
        for row in _read_csv(pair_path):
            try:
                row_fold = int(row["fold_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ComparisonError(f"SMRTnet prediction file has invalid fold at {pair_path}") from exc
            if row_fold != fold_id:
                raise ComparisonError(f"SMRTnet prediction file has a row for the wrong fold: {pair_path}")
            example_id = row.get("example_id", "")
            if not example_id:
                raise ComparisonError(f"SMRTnet prediction row is missing example_id: {pair_path}")
            label, probability = _binary_probability(
                row, source="SMRTnet", key="binding_probability"
            )
            if example_id in model_by_id:
                raise ComparisonError(f"duplicate SMRTnet test prediction for fold={fold_id}, {example_id}")
            model_by_id[example_id] = (label, probability, row)

        baseline = baseline_by_fold[fold_id]
        matched_ids = sorted(set(model_by_id) & set(baseline))
        if not matched_ids:
            raise ComparisonError(f"no matched eligible test IDs for fold {fold_id}")
        labels: list[int] = []
        model_scores: list[float] = []
        baseline_scores: list[float] = []
        for example_id in matched_ids:
            model_label, model_score, model_row = model_by_id[example_id]
            baseline_label, baseline_score, baseline_row = baseline[example_id]
            if model_label != baseline_label:
                raise ComparisonError(
                    f"label mismatch for fold={fold_id}, example={example_id}"
                )
            labels.append(model_label)
            model_scores.append(model_score)
            baseline_scores.append(baseline_score)
            matched_predictions.append(
                {
                    "fold_id": fold_id,
                    "example_id": example_id,
                    "aptamer_id": model_row.get("aptamer_id", baseline_row.get("aptamer_id", "")),
                    "molecule_id": model_row.get("molecule_id", baseline_row.get("molecule_id", "")),
                    "label": model_label,
                    "smrtnet_probability": model_score,
                    "thesis_a_probability": baseline_score,
                    "smrtnet_window_count": model_row.get("window_count", ""),
                }
            )
        fold_results[str(fold_id)] = {
            "strategy": strategy,
            "fold_id": fold_id,
            "thesis_a_test_rows": len(baseline),
            "smrtnet_eligible_test_rows": len(model_by_id),
            "matched_test_rows": len(matched_ids),
            "excluded_from_match": len(set(baseline) - set(model_by_id)),
            "smrtnet": _metric_payload(labels, model_scores),
            "thesis_a": _metric_payload(labels, baseline_scores),
        }

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(
        destination / "matched_predictions.csv",
        (
            "fold_id",
            "example_id",
            "aptamer_id",
            "molecule_id",
            "label",
            "smrtnet_probability",
            "thesis_a_probability",
            "smrtnet_window_count",
        ),
        matched_predictions,
    )
    result = {
        "status": "complete",
        "strategy": strategy,
        "run_id": summary.get("run_id"),
        "smrtnet_run_dir": str(run_dir),
        "thesis_a_predictions": str(baseline_path),
        "metric_source": "affinity_experiment.metrics.calculate_metrics",
        "pr_auc_definition": "average_precision_score",
        "threshold": 0.5,
        "folds": fold_results,
        "summary": _summary(fold_results),
        "note": "SMRTnet pair scores use the selected model for the matching held-out fold.",
    }
    with (destination / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(
        destination / "fold_metrics.csv",
        (
            "fold_id",
            "thesis_a_test_rows",
            "smrtnet_eligible_test_rows",
            "matched_test_rows",
            "excluded_from_match",
            "smrtnet_roc_auc",
            "thesis_a_roc_auc",
            "smrtnet_pr_auc",
            "thesis_a_pr_auc",
            "smrtnet_accuracy",
            "thesis_a_accuracy",
            "smrtnet_balanced_accuracy",
            "thesis_a_balanced_accuracy",
            "smrtnet_precision",
            "thesis_a_precision",
            "smrtnet_recall",
            "thesis_a_recall",
            "smrtnet_f1",
            "thesis_a_f1",
        ),
        [
            {
                "fold_id": value["fold_id"],
                "thesis_a_test_rows": value["thesis_a_test_rows"],
                "smrtnet_eligible_test_rows": value["smrtnet_eligible_test_rows"],
                "matched_test_rows": value["matched_test_rows"],
                "excluded_from_match": value["excluded_from_match"],
                **{
                    f"{model}_{metric}": value[model][metric]
                    for model in ("smrtnet", "thesis_a")
                    for metric in _METRICS
                },
            }
            for value in fold_results.values()
        ],
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smrtnet-run-dir", type=Path, required=True)
    parser.add_argument("--thesis-a-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = compare_predictions(
        args.smrtnet_run_dir,
        args.thesis_a_predictions,
        args.output_dir,
    )
    print(f"Wrote fold-matched comparison to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation
    raise SystemExit(main())
