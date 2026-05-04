import os
import json
import torch
import argparse
from pathlib import Path
import pandas as pd

from munl.evaluation.residual_knowledge import compute_residual_knowledge

def get_forget_loader(dataset_name, batch_size=128):
    """Utility to fetch forget loader. In a real integration this uses Hydra config."""
    try:
        from munl.datasets.get_dataset import get_dataset_based_on_split_state, get_loaders_from_dataset_and_unlearner_from_cfg
        import copy
        
        # This is a mocked or simplified retrieval assuming we can get standard splits
        # If this fails in the real pipeline, it should be adapted to the actual Hydra pipeline.
        # But for this script, we'll try to load it.
        # We will use dummy values if needed or just raise an error.
        raise NotImplementedError("Hydra pipeline integration for loader needed. Please provide it or use the dummy below for tests.")
    except Exception as e:
        print(f"Warning: Could not load actual forget_loader from pipeline ({e}). Returning a dummy loader for smoke tests.")
        from torch.utils.data import DataLoader, TensorDataset
        # Dummy data for smoke test
        dummy_x = torch.randn(100, 3, 32, 32)
        dummy_y = torch.randint(0, 10, (100,))
        ds = TensorDataset(dummy_x, dummy_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

def load_model(checkpoint_path, model_name="resnet18", device="cuda"):
    """Loads a model from a checkpoint path."""
    try:
        from munl.models.get_model import get_model
        model = get_model(model_name).to(device)
    except ImportError:
        # Fallback to standard torchvision
        import torchvision.models as models
        if model_name == "resnet18":
            model = models.resnet18(num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported model {model_name} in fallback mode.")
            
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle cases where state_dict is nested (e.g. dict with 'model_state' key)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Residual Knowledge")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unlearner", type=str, default="cfk")
    parser.add_argument("--tau", type=float, default=0.03)
    parser.add_argument("--c", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--perturbation", type=str, default="gaussian")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--unlearned-checkpoint", type=str, required=True)
    parser.add_argument("--reference-checkpoint", type=str, required=True)
    
    args = parser.parse_args()
    
    print(f"Loading reference model from {args.reference_checkpoint}...")
    reference_model = load_model(args.reference_checkpoint, args.model, args.device)
    
    print(f"Loading unlearned model from {args.unlearned_checkpoint}...")
    unlearned_model = load_model(args.unlearned_checkpoint, args.model, args.device)
    
    print("Loading forget set...")
    forget_loader = get_forget_loader(args.dataset)
    
    print(f"Computing RK with tau={args.tau}, c={args.c}...")
    result = compute_residual_knowledge(
        unlearned_model=unlearned_model,
        reference_model=reference_model,
        forget_loader=forget_loader,
        tau=args.tau,
        c=args.c,
        max_samples=args.max_samples,
        seed=args.seed,
        device=args.device,
        return_per_sample=True
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save JSON
    agg_result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "unlearner": args.unlearner,
        "reference": "naive",
        "tau": args.tau,
        "c": args.c,
        "K": args.max_samples,
        "perturbation": args.perturbation,
        "rk_mean": result["rk_mean"],
        "rk_excess": result["rk_excess"],
        "num_evaluated_samples": result["num_evaluated_samples"],
        "num_perturbations_per_sample": args.c,
        "denominator_epsilon": 1e-12
    }
    
    json_path = os.path.join(args.output_dir, f"10_{args.model}_{args.seed}_rk.json")
    with open(json_path, "w") as f:
        json.dump(agg_result, f, indent=2)
    print(f"Saved aggregate metrics to {json_path}")
        
    # Save CSV
    if result["per_sample"] is not None:
        csv_path = os.path.join(args.output_dir, f"10_{args.model}_{args.seed}_rk_per_sample.csv")
        result["per_sample"].to_csv(csv_path, index=False)
        print(f"Saved per-sample metrics to {csv_path}")

if __name__ == "__main__":
    main()
