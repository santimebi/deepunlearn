import argparse
import json
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RK Tables")
    parser.add_argument("--metrics-file", type=Path, default=Path("reports/raw_metrics.csv"), help="Path to raw_metrics.csv")
    parser.add_argument("--rk-artifacts-dir", type=Path, default=Path("artifacts/cifar10/residual_knowledge"), help="Path to RK JSON outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/resnet18_cifar10_rk"), help="Output directory")
    parser.add_argument("--baseline-method", type=str, default="naive", help="Method used as the gold standard")
    return parser.parse_args()

def compute_retention_deviation(row):
    return abs(row.get('TR', 1.0) - 1) + abs(row.get('RR', 1.0) - 1) + abs(row.get('FR', 1.0) - 1)

def main():
    args = parse_args()
    
    if not args.metrics_file.exists():
        print(f"Warning: {args.metrics_file} does not exist.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    df_raw = pd.read_csv(args.metrics_file)
    
    # Extract RK values
    rk_records = []
    if args.rk_artifacts_dir.exists():
        for json_path in args.rk_artifacts_dir.rglob("*_rk.json"):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    rk_records.append({
                        "method": data.get("unlearner", "unknown"),
                        "seed": data.get("seed", 0),
                        "RK@0.03": data.get("rk_mean", float('nan')),
                        "ExcessRK@0.03": data.get("rk_excess", float('nan'))
                    })
            except Exception as e:
                print(f"Failed to read {json_path}: {e}")
                
    df_rk = pd.DataFrame(rk_records)
    
    # Merge
    if not df_rk.empty:
        df_merged = pd.merge(df_raw, df_rk, on=["method", "seed"], how="left")
    else:
        df_merged = df_raw.copy()
        df_merged["RK@0.03"] = float('nan')
        df_merged["ExcessRK@0.03"] = float('nan')
        
    df_merged.to_csv(args.output_dir / "final_metrics_with_rk.csv", index=False)
    
    # Compute method means
    baseline_df = df_merged[df_merged['method'] == args.baseline_method]
    baseline_stats = {}
    if not baseline_df.empty:
        baseline_stats = baseline_df.mean(numeric_only=True).to_dict()
        
    df_methods = df_merged.groupby('method').mean(numeric_only=True).reset_index()
    
    df_methods['RR'] = df_methods['Retain'] / baseline_stats.get('Retain', 1.0)
    df_methods['FR'] = df_methods['Forget'] / baseline_stats.get('Forget', 1.0)
    df_methods['TR'] = df_methods['Test'] / baseline_stats.get('Test', 1.0)
    df_methods['RetDev'] = df_methods.apply(compute_retention_deviation, axis=1)
    
    if 'Test MIA' in df_methods.columns:
        df_methods['T-MIA'] = df_methods['Test MIA']
        df_methods['Indisc'] = 1.0 - abs(2 * df_methods['T-MIA'] - 1.0)
    else:
        df_methods['Indisc'] = 0.0

    # 1. Ranking RK (minimize ExcessRK)
    # Filter methods that have RK
    df_rk_ranking = df_methods[['method', 'RK@0.03', 'ExcessRK@0.03']].copy()
    df_rk_ranking = df_rk_ranking.dropna(subset=['ExcessRK@0.03'])
    df_rk_ranking.sort_values(by='ExcessRK@0.03', ascending=True, inplace=True)
    df_rk_ranking['Ranking'] = df_rk_ranking['ExcessRK@0.03'].rank(method='min').astype(int)
    
    cols = ['Ranking', 'method', 'ExcessRK@0.03', 'RK@0.03']
    df_rk_ranking[cols].to_csv(args.output_dir / "ranking_rk.csv", index=False)
    
    # 2. Ranking RetDev/Indisc + RK
    df_combined = df_methods[['method', 'RetDev', 'Indisc', 'RK@0.03', 'ExcessRK@0.03']].copy()
    df_combined.sort_values(by=['RetDev', 'Indisc'], ascending=[True, False], inplace=True)
    df_combined['Ranking'] = df_combined['RetDev'].rank(method='min').astype(int)
    
    cols_combined = ['Ranking', 'method', 'RetDev', 'Indisc', 'RK@0.03', 'ExcessRK@0.03']
    df_combined[cols_combined].to_csv(args.output_dir / "ranking_retdev_indisc_rk.csv", index=False)
    
    print(f"Generated tables in {args.output_dir}")

if __name__ == "__main__":
    main()
