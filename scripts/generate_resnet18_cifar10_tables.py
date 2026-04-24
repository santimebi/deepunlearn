#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import traceback

import pandas as pd
import torch

from munl.configurations import get_img_size_for_dataset, get_num_classes
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg
from munl.evaluation.accuracy import compute_accuracy
from munl.evaluation.common import extract_predictions
from munl.evaluation.membership_inference_attack import evaluate_mia_on_model
from munl.models import get_model
from munl.settings import default_evaluation_loaders
from munl.utils import DictConfig, setup_seed


SEED_RE = re.compile(r"10_resnet18_(\d+)\.pth$")
TIME_RE = re.compile(r"^time:\s*([0-9eE+\-.]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Candidate:
    method: str
    seed: int
    model_path: Path
    meta_path: Path
    source_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18+CIFAR10 final models and generate reports/raw_metrics.csv"
    )
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_csv_set(value: str | None, cast=int):
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


def parse_runtime_seconds(meta_path: Path):
    if not meta_path.exists():
        return None
    text = meta_path.read_text(encoding="utf-8", errors="ignore")
    m = TIME_RE.search(text)
    return float(m.group(1)) if m else None


def collect_dir_candidates(method: str, directory: Path, source_group: str, seeds_filter):
    if not directory.exists():
        return []
    out = []
    for model_path in sorted(directory.glob("10_resnet18_*.pth")):
        seed = parse_seed_from_name(model_path)
        if seeds_filter is not None and seed not in seeds_filter:
            continue
        meta_path = model_path.with_name(model_path.stem + "_meta.txt")
        out.append(Candidate(method, seed, model_path, meta_path, source_group))
    return out


def collect_candidates(artifacts_root: Path, methods_filter, seeds_filter, include_original: bool):
    candidates = []

    naive_dir = artifacts_root / "cifar10" / "unlearn" / "unlearner_naive"
    original_dir = artifacts_root / "cifar10" / "unlearn" / "unlearner_original"
    objective_root = artifacts_root / "cifar10" / "objective10"

    if methods_filter is None or "naive" in methods_filter:
        candidates.extend(collect_dir_candidates("naive", naive_dir, "unlearn", seeds_filter))

    if include_original and (methods_filter is None or "original" in methods_filter):
        candidates.extend(collect_dir_candidates("original", original_dir, "unlearn", seeds_filter))

    if objective_root.exists():
        for method_dir in sorted(objective_root.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            if methods_filter is not None and method not in methods_filter:
                continue
            candidates.extend(collect_dir_candidates(method, method_dir, "objective10", seeds_filter))

    return sorted(candidates, key=lambda x: (x.method, x.seed))


def build_eval_loaders(batch_size: int, random_state: int):
    setup_seed(random_state)
    dataset_cfg = DictConfig({"name": "cifar10"})
    unlearner_cfg = DictConfig(
        {"loaders": default_evaluation_loaders(), "batch_size": batch_size}
    )
    _, retain_loader, forget_loader, val_loader, test_loader = (
        get_loaders_from_dataset_and_unlearner_from_cfg(
            root=Path("."),
            dataset_cfg=dataset_cfg,
            unlearner_cfg=unlearner_cfg,
            random_state=random_state,
        )
    )
    return retain_loader, forget_loader, val_loader, test_loader


def load_model(model_path: Path, device: str):
    num_classes = get_num_classes("cifar10")
    img_size = get_img_size_for_dataset("cifar10")
    model = get_model("resnet18", num_classes, img_size)
    weights = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    return model


def safe_mean(x):
    try:
        return float(x.mean())
    except Exception:
        return float("nan")


def evaluate_candidate(candidate: Candidate, device: str, loaders):
    retain_loader, forget_loader, val_loader, test_loader = loaders

    result = {
        "dataset": "cifar10",
        "model": "resnet18",
        "objective": candidate.source_group,
        "method": candidate.method,
        "seed": candidate.seed,
        "model_path": str(candidate.model_path),
        "meta_path": str(candidate.meta_path),
        "runtime_seconds": parse_runtime_seconds(candidate.meta_path),
        "Retain": float("nan"),
        "Forget": float("nan"),
        "Val": float("nan"),
        "Test": float("nan"),
        "Val MIA": float("nan"),
        "Test MIA": float("nan"),
        "status": "unknown",
        "error_message": "",
    }

    try:
        model = load_model(candidate.model_path, device=device)

        for metric_name, loader in [
            ("Retain", retain_loader),
            ("Forget", forget_loader),
            ("Val", val_loader),
            ("Test", test_loader),
        ]:
            y_true, y_pred = extract_predictions(model, loader, device=device)
            result[metric_name] = float(compute_accuracy(y_true, y_pred))

        mia_errors = []

        try:
            result["Val MIA"] = safe_mean(evaluate_mia_on_model(model, val_loader, forget_loader))
        except Exception as e:
            mia_errors.append(f"Val MIA failed: {type(e).__name__}: {e}")

        try:
            result["Test MIA"] = safe_mean(evaluate_mia_on_model(model, test_loader, forget_loader))
        except Exception as e:
            mia_errors.append(f"Test MIA failed: {type(e).__name__}: {e}")

        if mia_errors:
            result["status"] = "partial_mia_failure"
            result["error_message"] = " | ".join(mia_errors)
        else:
            result["status"] = "ok"

    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = f"{type(e).__name__}: {e}"

    return result


def main():
    args = parse_args()
    methods_filter = parse_csv_set(args.methods, cast=str)
    seeds_filter = parse_csv_set(args.seeds, cast=int)

    artifacts_root = args.artifacts_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics_path = output_dir / "raw_metrics.csv"

    existing_rows = []
    done_pairs = set()

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

    print("[setup] Building shared evaluation loaders once...")
    loaders = build_eval_loaders(args.batch_size, args.random_state)

    rows = list(existing_rows)
    total = len(candidates)
    processed = 0

    for candidate in candidates:
        key = (candidate.method, candidate.seed)
        if key in done_pairs:
            print(f"[skip] method={candidate.method} seed={candidate.seed}")
            continue

        processed += 1
        print(f"[{processed}/{total}] Evaluating method={candidate.method} seed={candidate.seed} path={candidate.model_path}")

        result = evaluate_candidate(candidate, args.device, loaders)
        rows.append(result)

        df_partial = pd.DataFrame(rows)
        df_partial.sort_values(by=["method", "seed"], inplace=True)
        df_partial.to_csv(raw_metrics_path, index=False)
        print(f"[saved] {raw_metrics_path} | status={result['status']}")

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
