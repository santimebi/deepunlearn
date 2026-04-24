import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Generate Paper Tables (CSV + LaTeX)")
    parser.add_argument("--metrics-file", type=str, default="reports/raw_metrics.csv", help="Path to raw_metrics.csv")
    parser.add_argument("--baseline-method", type=str, default="naive", help="Method used as the gold standard (retrain/naive)")
    parser.add_argument("--output-dir", type=str, default="reports/paper_tables", help="Output directory for tables")
    return parser

def compute_retention_deviation(row):
    return abs(row['TR'] - 1) + abs(row['RR'] - 1) + abs(row['FR'] - 1)

def main():
    args = get_parser().parse_args()
    input_file = Path(args.metrics_file)
    output_dir = Path(args.output_dir)
    baseline_method = args.baseline_method

    if not input_file.exists():
        print(f"Warning: {input_file} does not exist. Please run the evaluation pipeline first.")
        # We exit cleanly allowing the run gracefully later
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(input_file)

    # Calculate basic accuracies (Assuming Retain, Forget, Test columns exist as accuracy [0, 1])
    # The paper uses percentages or ratios. We'll ensure they are ratios for Retention calculation.
    
    # 1. Extract Baseline Performance (Means over seeds)
    baseline_df = df_raw[df_raw['method'] == baseline_method]
    if baseline_df.empty:
         print(f"Warning: Baseline method '{baseline_method}' not found. Cannot compute retention or RTE accurately.")
         baseline_stats = {'Retain': 1.0, 'Forget': 1.0, 'Test': 1.0, 'runtime_seconds': 1.0}
    else:
         baseline_stats = baseline_df.mean(numeric_only=True).to_dict()

    # 2. Compute Method Aggregates (Mean across seeds)
    df_methods = df_raw.groupby('method').mean(numeric_only=True).reset_index()

    # Calculate Retention Metrics
    # RA: Retain Accuracy, FA: Forget Accuracy, TA: Test Accuracy
    df_methods['RA'] = df_methods['Retain']
    df_methods['FA'] = df_methods['Forget']
    df_methods['TA'] = df_methods['Test']
    
    # RR, FR, TR
    df_methods['RR'] = df_methods['RA'] / baseline_stats.get('Retain', 1.0)
    df_methods['FR'] = df_methods['FA'] / baseline_stats.get('Forget', 1.0)
    df_methods['TR'] = df_methods['TA'] / baseline_stats.get('Test', 1.0)

    # RetDev
    df_methods['RetDev'] = df_methods.apply(compute_retention_deviation, axis=1)

    # Indisc (assuming 'indiscernibility' or 'Test MIA'/'Val MIA' is mapped)
    # The paper indiscernibility is Indisc = 1 - abs(2 * MIA - 1)
    if 'Test MIA' in df_methods.columns:
        df_methods['T-MIA'] = df_methods['Test MIA']
        df_methods['Indisc'] = 1.0 - abs(2 * df_methods['T-MIA'] - 1.0)
    elif 'indiscernibility' in df_methods.columns:
        df_methods['Indisc'] = df_methods['indiscernibility']
        df_methods['T-MIA'] = 0.5 # Placeholder if missing
    else:
        df_methods['Indisc'] = 0.0
        df_methods['T-MIA'] = 0.5

    # Run Time Efficiency (RTE)
    if 'runtime_seconds' in df_methods.columns:
        df_methods['RTE'] = baseline_stats.get('runtime_seconds', 1.0) / df_methods['runtime_seconds']
    else:
        df_methods['RTE'] = 1.0

    # ---------------------------------------------------------
    # GENERATE TABLES
    # ---------------------------------------------------------
    def export_dual(df, filename):
        # We explicitly round numeric types for clean output
        df = df.round(4)
        df.to_csv(output_dir / f"{filename}.csv")
        styled = df.style.format(precision=4)
        try:
            styled = styled.hide(axis='index')
        except Exception:
            try:
                styled = styled.hide_index()
            except Exception:
                pass
        styled.to_latex(
            output_dir / f"{filename}.tex",
            column_format='l' + 'c' * (len(df.columns) - 1),
            hrules=True 
        )

    # Table 1: Main Results (Method, RetDev, Indisc)
    table1 = df_methods[['method', 'RetDev', 'Indisc']].sort_values(by=['RetDev', 'Indisc'], ascending=[True, False])
    export_dual(table1, 'table1_main_results')

    # Table 2: Run-time efficiency
    table2 = df_methods[['method', 'RTE']].sort_values(by='RTE', ascending=False)
    export_dual(table2, 'table2_runtime_efficiency')

    # Table 6: Detailed metrics for ResNet18 + CIFAR10
    cols_t6 = ['method', 'RA', 'FA', 'TA', 'RR', 'FR', 'TR', 'RetDev', 'Indisc', 'T-MIA', 'RTE']
    # Select available columns
    avail_cols = [c for c in cols_t6 if c in df_methods.columns]
    table6 = df_methods[avail_cols]
    export_dual(table6, 'table6_detailed_metrics')

    # Table 13: Rankings (By RetDev as primary, Indisc as secondary for ResNet18)
    table13 = table1.copy()
    table13['Ranking'] = table13['RetDev'].rank(method='min').astype(int)
    # Reorder columns
    table13 = table13[['Ranking', 'method', 'RetDev', 'Indisc']]
    export_dual(table13, 'table13_architecture_rankings')

    print(f"Successfully generated dual exports for Tables 1, 2, 6, and 13 in '{output_dir}'.")

if __name__ == "__main__":
    main()
