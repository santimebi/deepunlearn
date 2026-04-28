import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from munl.evaluation.residual_knowledge import compute_residual_knowledge

class DummyModel(torch.nn.Module):
    def __init__(self, mode="always_correct"):
        super().__init__()
        self.mode = mode
        
    def forward(self, x):
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, 10, device=x.device)
        
        if self.mode == "always_correct":
            # For simplicity, let's assume true label is 0 in tests
            logits[:, 0] = 100.0
        elif self.mode == "half_correct":
            # Correct half the time (alternating)
            for i in range(batch_size):
                if i % 2 == 0:
                    logits[i, 0] = 100.0
                else:
                    logits[i, 1] = 100.0
        elif self.mode == "never_correct":
            logits[:, 1] = 100.0
            
        return logits

@pytest.fixture
def dummy_loader():
    x = torch.zeros(10, 3, 32, 32)
    y = torch.zeros(10, dtype=torch.long)  # True labels are all 0
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=5)

def test_identical_models_rk_approx_1(dummy_loader):
    model1 = DummyModel("half_correct")
    model2 = DummyModel("half_correct")
    
    res = compute_residual_knowledge(
        model1, model2, dummy_loader, 
        tau=0.03, c=10, device="cpu"
    )
    
    assert res["rk_mean"] == pytest.approx(1.0, 1e-4)

def test_unlearned_better_than_reference_rk_greater_than_1(dummy_loader):
    unlearned = DummyModel("always_correct")
    reference = DummyModel("half_correct")
    
    res = compute_residual_knowledge(
        unlearned, reference, dummy_loader, 
        tau=0.03, c=10, device="cpu"
    )
    
    assert res["rk_mean"] > 1.0

def test_denominator_zero_does_not_crash(dummy_loader):
    unlearned = DummyModel("always_correct")
    reference = DummyModel("never_correct")
    
    res = compute_residual_knowledge(
        unlearned, reference, dummy_loader, 
        tau=0.03, c=10, device="cpu", denominator_epsilon=1e-12
    )
    
    # Since denominator is 0, it uses epsilon 1e-12, meaning RK should be very large
    assert res["rk_mean"] > 1000.0
    
def test_fixed_seed_deterministic(dummy_loader):
    unlearned = DummyModel("half_correct")
    reference = DummyModel("half_correct")
    
    res1 = compute_residual_knowledge(
        unlearned, reference, dummy_loader, 
        tau=0.1, c=10, seed=42, device="cpu", return_per_sample=True
    )
    
    res2 = compute_residual_knowledge(
        unlearned, reference, dummy_loader, 
        tau=0.1, c=10, seed=42, device="cpu", return_per_sample=True
    )
    
    assert res1["rk_mean"] == res2["rk_mean"]
    assert res1["rk_excess"] == res2["rk_excess"]
