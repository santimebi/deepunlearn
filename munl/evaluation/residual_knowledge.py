import torch
import pandas as pd
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader

# Optional dataset import for correct normalization if available
try:
    from munl.datasets.cifar10 import CIFAR10_MEAN, CIFAR10_STD
except ImportError:
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def gaussian_perturb(x: torch.Tensor, tau: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Applies Gaussian perturbations to CIFAR-10 normalized tensors using Option B:
    Denormalize -> Add Noise -> Clamp [0,1] -> Renormalize
    """
    device = x.device
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    
    # Denormalize
    x_pixel = x * std + mean
    
    # Add noise
    noise = torch.randn(x_pixel.size(), generator=generator, dtype=x_pixel.dtype, device=x_pixel.device) * tau
    x_perturbed_pixel = x_pixel + noise
    
    # Clamp to [0, 1] valid pixel domain
    x_perturbed_pixel = torch.clamp(x_perturbed_pixel, 0.0, 1.0)
    
    # Renormalize
    x_perturbed = (x_perturbed_pixel - mean) / std
    return x_perturbed

def compute_residual_knowledge(
    unlearned_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    forget_loader: DataLoader,
    *,
    tau: float = 0.03,
    c: int = 100,
    max_samples: Optional[int] = None,
    seed: int = 123,
    device: str = "cuda",
    denominator_epsilon: float = 1e-12,
    return_per_sample: bool = True,
) -> Dict[str, Any]:
    unlearned_model.eval()
    reference_model.eval()
    
    # Setup generator for deterministic perturbations
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    sample_records = []
    samples_processed = 0
    
    for x, y in forget_loader:
        if max_samples is not None and samples_processed >= max_samples:
            break
            
        B = x.size(0)
        
        # Limit batch size if max_samples restricts it
        if max_samples is not None and samples_processed + B > max_samples:
            B = max_samples - samples_processed
            x = x[:B]
            y = y[:B]
            
        x, y = x.to(device), y.to(device)
        
        # Repeat each sample c times
        # x shape: [B, C, H, W] -> [B, c, C, H, W] -> [B*c, C, H, W]
        x_rep = x.unsqueeze(1).repeat(1, c, 1, 1, 1).view(-1, *x.shape[1:])
        y_rep = y.unsqueeze(1).repeat(1, c).view(-1)
        
        # Perturb
        x_pert = gaussian_perturb(x_rep, tau, generator)
        
        with torch.no_grad():
            pred_unlearned = unlearned_model(x_pert).argmax(dim=1)
            pred_reference = reference_model(x_pert).argmax(dim=1)
            
        # Reshape to [B, c] and count corrects
        correct_unlearned = (pred_unlearned == y_rep).reshape(B, c).sum(dim=1).float()
        correct_reference = (pred_reference == y_rep).reshape(B, c).sum(dim=1).float()
        
        for i in range(B):
            u_correct = correct_unlearned[i].item()
            r_correct = correct_reference[i].item()
            u_rate = u_correct / c
            r_rate = r_correct / c
            
            rk_val = u_rate / max(r_rate, denominator_epsilon)
            
            sample_records.append({
                "sample_index": samples_processed + i,
                "label": y[i].item(),
                "tau": tau,
                "c": c,
                "unlearned_correct_count": int(u_correct),
                "reference_correct_count": int(r_correct),
                "unlearned_correct_rate": u_rate,
                "reference_correct_rate": r_rate,
                "rk_sample": rk_val,
                "rk_sample_excess": max(0.0, rk_val - 1.0)
            })
            
        samples_processed += B
        
    df = pd.DataFrame(sample_records)
    
    if len(df) == 0:
        return {
            "rk_mean": 0.0,
            "rk_excess": 0.0,
            "tau": tau,
            "c": c,
            "max_samples": max_samples,
            "num_evaluated_samples": 0,
            "per_sample": None,
        }
        
    rk_mean = df["rk_sample"].mean()
    rk_excess = df["rk_sample_excess"].mean()
    
    result = {
        "rk_mean": float(rk_mean),
        "rk_excess": float(rk_excess),
        "tau": tau,
        "c": c,
        "max_samples": max_samples,
        "num_evaluated_samples": samples_processed,
        "per_sample": df if return_per_sample else None,
    }
    
    return result
