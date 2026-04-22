import argparse
import json
import warnings
from pathlib import Path
from math import sqrt
import pandas as pd
from datetime import datetime

from munl.evaluation.evaluate_model import ModelEvaluationFromPathApp
from munl.evaluation import indiscernibility

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--objective", type=str, default="objective10")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--artifacts-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--skip-existing", action="store_true")
    return parser

def parse_runtime_from_meta(meta_path: Path):
    if not meta_path.exists():
        return float('nan')
    with open(meta_path, 'r', encoding='utf-8') as f:
        # En caso de no tener un JSON estándar, iteramos las líneas buscando "time" o "runtime"
        for line in f:
            if "runtime" in line.lower() or "time" in line.lower():
                try:
                    # Intenta extraer el número al final de la línea: Ej. "Runtime: 45.2"
                    return float(line.split()[-1]) 
                except:
                    pass
    return float('nan')

def main():
    args = get_parser().parse_args()
    artifacts_root = Path(args.artifacts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    naive_dir = artifacts_root / args.dataset / "unlearn" / "unlearner_naive"
    methods_dir = artifacts_root / args.dataset / args.objective
    
    # 0. Descubrir métodos
    if not methods_dir.exists():
        raise FileNotFoundError(f"Critical Error: Methods directory not found at {methods_dir}")
    
    if args.methods == "all":
        methods = [d.name for d in methods_dir.iterdir() if d.is_dir()]
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        
    failed_methods = {}
    valid_methods = []
    
    # 1. Validación Estructural Naive
    for seed in seeds:
        naive_model = naive_dir / f"10_{args.model}_{seed}.pth"
        if not naive_model.exists():
            raise FileNotFoundError(f"CRITICAL: Naive baseline model missing for seed {seed}. Expected path: {naive_model}")
            
    # 2. Validación Estructural Métodos (Tolerancia a fallos parciales)
    for method in methods:
        method_path = methods_dir / method
        is_valid = True
        for seed in seeds:
            m_model = method_path / f"10_{args.model}_{seed}.pth"
            m_meta = method_path / f"10_{args.model}_{seed}_meta.txt"
            if not m_model.exists():
                failed_methods[method] = f"Missing model file: {m_model}"
                is_valid = False
                break
            if not m_meta.exists():
                failed_methods[method] = f"Missing metadata file: {m_meta}"
                is_valid = False
                break
        if is_valid:
            valid_methods.append(method)
            
    if failed_methods:
        warnings.warn(f"WARNING: Graceful degradation activated. Some methods were excluded due to missing artifacts: {list(failed_methods.keys())}")
        
    # 3. Evaluación Naive (con caché)
    naive_cache_file = output_dir / "naive_baseline_cache.csv"
    if naive_cache_file.exists():
        print(f"Loading Naive baseline evaluations from cache: {naive_cache_file}")
        df_naive = pd.read_csv(naive_cache_file)
    else:
        print("Evaluating Naive baseline from scratch...")
        naive_rows = []
        for seed in seeds:
            naive_model = naive_dir / f"10_{args.model}_{seed}.pth"
            app = ModelEvaluationFromPathApp(
                model_path=str(naive_model),
                model_type=args.model,
                dataset=args.dataset,
                batch_size=args.batch_size,
                random_state=seed,
                device=args.device
            )
            df_eval = app.run()
            row = df_eval.iloc[0].to_dict()
            row["method"] = "naive"
            row["seed"] = seed
            naive_rows.append(row)
        df_naive = pd.DataFrame(naive_rows)
        df_naive.to_csv(naive_cache_file, index=False)
        
    naive_dict = df_naive.set_index("seed").to_dict('index')

    # 4. Evaluación de Métodos
    raw_rows = []
    for method in valid_methods:
        print(f"Evaluating method: {method}")
        method_path = methods_dir / method
        for seed in seeds:
            m_model = method_path / f"10_{args.model}_{seed}.pth"
            m_meta = method_path / f"10_{args.model}_{seed}_meta.txt"
            
            app = ModelEvaluationFromPathApp(
                model_path=str(m_model),
                model_type=args.model,
                dataset=args.dataset,
                batch_size=args.batch_size,
                random_state=seed,
                device=args.device
            )
            df_eval = app.run()
            row = df_eval.iloc[0].to_dict()
            row["method"] = method
            row["seed"] = seed
            row["model_path"] = str(m_model)
            row["runtime_seconds"] = parse_runtime_from_meta(m_meta)
            
            # Derived Metrics & Ranking Formula (Heredado de Optuna y Functional Doc)
            n_row = naive_dict[seed]
            retain_loss = abs(row["Retain"] - n_row["Retain"])
            forget_loss = abs(row["Forget"] - n_row["Forget"])
            val_loss = abs(row["Val"] - n_row["Val"])
            indis = indiscernibility(row["Val MIA"])
            dis = 1.0 - indis
            
            row["retain_loss_vs_naive"] = retain_loss
            row["forget_loss_vs_naive"] = forget_loss
            row["val_loss_vs_naive"] = val_loss
            row["discernibility"] = dis
            row["indiscernibility"] = indis
            
            # L2 Norm con pesos asimétricos
            obj_score = sqrt(
                ((1/3) * retain_loss)**2 + 
                ((1/3) * forget_loss)**2 + 
                ((1/3) * val_loss)**2 + 
                (1.0 * dis)**2
            )
            row["objective10_score"] = obj_score
            raw_rows.append(row)

    # 5. Agrupación y exportación de CSVs y Markdowns
    if raw_rows:
        df_raw = pd.DataFrame(raw_rows)
        # Ordenación base de raw metrics
        cols = ["method", "seed", "model_path", "Retain", "Forget", "Val", "Test", "Val MIA", "Test MIA", "runtime_seconds"]
        df_raw_base = df_raw[cols + [c for c in df_raw.columns if c not in cols]]
        df_raw_base.to_csv(output_dir / "raw_metrics.csv", index=False)
        df_raw_base.to_markdown(output_dir / "raw_metrics.md", index=False)
        
        # Sub-tabla: Ranking a nivel seed
        ranking_cols = ["method", "seed", "retain_loss_vs_naive", "forget_loss_vs_naive", 
                        "val_loss_vs_naive", "Val MIA", "discernibility", "indiscernibility", 
                        "objective10_score"]
        df_ranking = df_raw[ranking_cols]
        df_ranking.to_csv(output_dir / "ranking_seed_level.csv", index=False)
        
        # Tabla Agregada (Summary)
        agg_cols = ["Retain", "Forget", "Val", "Test", "Val MIA", "Test MIA", "runtime_seconds", "objective10_score"]
        df_agg = df_raw.groupby("method")[agg_cols].agg(['mean', 'std']).reset_index()
        # Aplanar MultiIndex
        df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
        
        df_agg.to_csv(output_dir / "method_summary.csv", index=False)
        df_agg.to_markdown(output_dir / "method_summary.md", index=False)
        
        # Ranking Final
        df_final = df_agg.sort_values(by="objective10_score_mean", ascending=True).reset_index(drop=True)
        df_final.index += 1
        df_final.insert(0, "ranking", df_final.index) # Add ranking column as first
        df_final.to_csv(output_dir / "ranking_final.csv", index=False)
        df_final.to_markdown(output_dir / "ranking_final.md", index=False)
    else:
        print("No valid methods evaluated. No tables generated.")

    # 6. Serialización del Manifest para Trazabilidad
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "seeds_evaluated": seeds,
        "valid_methods_count": len(valid_methods),
        "valid_methods": valid_methods,
        "failed_methods": failed_methods,
        "artifacts_root": str(artifacts_root),
        "output_dir": str(output_dir)
    }
    with open(output_dir / "run_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Post-process report generation complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
