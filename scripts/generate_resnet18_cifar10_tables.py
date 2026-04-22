#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from munl.evaluation.evaluate_model import ModelEvaluationFromPathApp


SEED_RE = re.compile(r"10_resnet18_(\d+)\.pth$")
TIME_RE = re.compile(r"^time:\s*([0-9eE+\-.]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Candidate:
    method: str
    seed: int
    model_path: Path
    meta_path: Path
    source_group: str  # "objective10" | "unlearn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18+CIFAR10 final models and generate reports/raw_metrics.csv"
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=True,
        help="Path to artifacts root, e.g. /data/santiago.medina/deepunlearn/artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where raw_metrics.csv will be written",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random state passed to evaluation pipeline",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods to evaluate. Supports naive, original and objective10 methods. Example: naive,cfk,catastrophic_forgetting_gamma_k",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to evaluate. Example: 0,1,2",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Also evaluate unlearner_original models",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If reports/raw_metrics.csv exists, skip already evaluated (method, seed) pairs",
    )
    return parser.parse_args()


def parse_csv_set(value: str | None, cast=int) -> set | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return {cast(x.strip()) for x in value.split(",") if x.strip()}


def parse_seed_from_name(path: Path) -> int:
    m = SEED_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint name: {path}")
    return int(m.group(1))


def parse_runtime_seconds(meta_path: Path) -> float | None:
    if not meta_path.exists():
        return None
    text = meta_path.read_text(encoding="utf-8", errors="ignore")
    m = TIME_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def collect_dir_candidates(
    method: str,
    directory: Path,
    source_group: str,
    seeds_filter: set[int] | None,
) -> list[Candidate]:
    if not directory.exists():
        return []

    out: list[Candidate] = []
    for model_path in sorted(directory.glob("10_resnet18_*.pth")):
        seed = parse_seed_from_name(model_path)
        if seeds_filter is not None and seed not in seeds_filter:
            continue
        meta_path = model_path.with_name(model_path.stem + "_meta.txt")
        out.append(
            Candidate(
                method=method,
                seed=seed,
                model_path=model_path,
                meta_path=meta_path,
                source_group=source_group,
            )
        )
    return out


def collect_candidates(
    artifacts_root: Path,
    methods_filter: set[str] | None,
    seeds_filter: set[int] | None,
    include_original: bool,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    naive_dir = artifacts_root / "cifar10" / "unlearn" / "unlearner_naive"
    original_dir = artifacts_root / "cifar10" / "unlearn" / "unlearner_original"
    objective_root = artifacts_root / "cifar10" / "objective10"

    if methods_filter is None or "naive" in methods_filter:
        candidates.extend(
            collect_dir_candidates(
                method="naive",
                directory=naive_dir,
                source_group="unlearn",
                seeds_filter=seeds_filter,
            )
        )

    if include_original and (methods_filter is None or "original" in methods_filter):
        candidates.extend(
            collect_dir_candidates(
                method="original",
                directory=original_dir,
                source_group="unlearn",
                seeds_filter=seeds_filter,
            )
        )

    if objective_root.exists():
        for method_dir in sorted(objective_root.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if methods_filter is not None and method not in methods_filter:
                continue
            candidates.extend(
                collect_dir_candidates(
                    method=method,
                    directory=method_dir,
                    source_group="objective10",
                    seeds_filter=seeds_filter,
                )
            )

    return sorted(candidates, key=lambda x: (x.method, x.seed))


def evaluate_candidate(
    candidate: Candidate,
    device: str,
    batch_size: int,
    random_state: int,
) -> dict:
    app = ModelEvaluationFromPathApp(
        model_path=str(candidate.model_path),
        model_type="resnet18",
        dataset="cifar10",
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )
    df = app.run()
    row = df.iloc[0].to_dict()

    return {
        "dataset": "cifar10",
        "model": "resnet18",
        "objective": candidate.source_group,
        "method": candidate.method,
        "seed": candidate.seed,
        "model_path": str(candidate.model_path),
        "meta_path": str(candidate.meta_path),
        "runtime_seconds": parse_runtime_seconds(candidate.meta_path),
        **row,
    }


def main() -> None:
    args = parse_args()

    methods_filter = parse_csv_set(args.methods, cast=str)
    seeds_filter = parse_csv_set(args.seeds, cast=int)

    artifacts_root = args.artifacts_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics_path = output_dir / "raw_metrics.csv"

    existing_rows: list[dict] = []
    done_pairs: set[tuple[str, int]] = set()

    if args.resume and raw_metrics_path.exists():
        existing_df = pd.read_csv(raw_metrics_path)
        existing_rows = existing_df.to_dict(orient="records")
        if {"method", "seed"}.issubset(existing_df.columns):
            done_pairs = {
                (str(r["method"]), int(r["seed"]))
                for _, r in existing_df[["method", "seed"]].iterrows()
            }

    candidates = collect_candidates(
        artifacts_root=artifacts_root,
        methods_filter=methods_filter,
        seeds_filter=seeds_filter,
        include_original=args.include_original,
    )

    if not candidates:
        raise SystemExit("No candidate models found with the provided filters.")

    rows = list(existing_rows)
    total = len(candidates)
    processed = 0

    for candidate in candidates:
        key = (candidate.method, candidate.seed)
        if key in done_pairs:
            print(f"[skip] method={candidate.method} seed={candidate.seed} already present in raw_metrics.csv")
            continue

        processed += 1
        print(
            f"[{processed}/{total}] Evaluating method={candidate.method} seed={candidate.seed} "
            f"path={candidate.model_path}"
        )

        result = evaluate_candidate(
            candidate=candidate,
            device=args.device,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        rows.append(result)

        df_partial = pd.DataFrame(rows)
        df_partial.sort_values(by=["method", "seed"], inplace=True)
        df_partial.to_csv(raw_metrics_path, index=False)
        print(f"[saved] {raw_metrics_path}")

    df_final = pd.read_csv(raw_metrics_path)
    df_final.sort_values(by=["method", "seed"], inplace=True)
    df_final.to_csv(raw_metrics_path, index=False)

    means_path = output_dir / "raw_metrics_method_means.csv"
    df_means = df_final.groupby("method").mean(numeric_only=True).reset_index()
    df_means.to_csv(means_path, index=False)

    print(f"[done] raw metrics: {raw_metrics_path}")
    print(f"[done] grouped means: {means_path}")


if __name__ == "__main__":
    main()