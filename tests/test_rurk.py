import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from omegaconf import OmegaConf

from munl.unlearning.rurk import RURKUnlearner, DefaultRURKConfig
from munl.configurations import unlearner_store

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def test_rurk_registered_in_store():
    # Verify RURK is registered in the unlearner store
    assert "rurk" in unlearner_store["unlearner"]

def test_rurk_instantiation():
    cfg = OmegaConf.structured(DefaultRURKConfig(epochs=1))
    unlearner = RURKUnlearner(cfg, device="cpu")
    assert unlearner is not None

def test_rurk_model_updates_and_no_frozen_layers():
    # Create simple data
    torch.manual_seed(42)
    x_r = torch.randn(10, 10)
    y_r = torch.randint(0, 2, (10,))
    x_f = torch.randn(5, 10)
    y_f = torch.randint(0, 2, (5,))
    
    retain_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=2)
    forget_loader = DataLoader(TensorDataset(x_f, y_f), batch_size=2)
    val_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=2)
    
    model = DummyModel()
    # Save initial weights
    initial_weights = model.fc.weight.clone()
    
    # Check no layers are frozen
    for param in model.parameters():
        assert param.requires_grad == True
    
    # Set up config and unlearner
    cfg = OmegaConf.structured(DefaultRURKConfig(epochs=1, learning_rate=0.1, tau=0.01))
    unlearner = RURKUnlearner(cfg, device="cpu")
    
    # Run unlearning
    updated_model = unlearner.unlearn(model, retain_loader, forget_loader, val_loader)
    
    # Verify weights changed
    assert not torch.allclose(initial_weights, updated_model.fc.weight)
    
    # Verify still no layers are frozen
    for param in updated_model.parameters():
        assert param.requires_grad == True
