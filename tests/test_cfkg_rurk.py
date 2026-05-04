import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

from munl.unlearning.cfkg_rurk import CFGKRURKUnlearner, DefaultCFGKRURKConfig
from munl.configurations import unlearner_store

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the expected input of gaussian_perturb (channels, height, width)
        # We will flatten it in the linear layer
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16*32*32, 2)
        
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

def test_cfkg_rurk_registered_in_store():
    assert ("unlearner", "cfkg_rurk") in unlearner_store

def test_cfkg_rurk_instantiation():
    cfg = OmegaConf.structured(DefaultCFGKRURKConfig())
    unlearner = CFGKRURKUnlearner(cfg, device="cpu")
    assert unlearner is not None
    assert "cfkg" in cfg
    assert "rurk" in cfg

def test_cfkg_rurk_model_execution():
    # CFKG+RURK is heavy so we just test that it runs without crashing 
    # on dummy data
    torch.manual_seed(42)
    # CIFAR-10 shape
    x_r = torch.randn(10, 3, 32, 32)
    y_r = torch.randint(0, 2, (10,))
    x_f = torch.randn(4, 3, 32, 32)
    y_f = torch.randint(0, 2, (4,))
    
    retain_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=2)
    forget_loader = DataLoader(TensorDataset(x_f, y_f), batch_size=2)
    val_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=2)
    
    model = DummyModel()
    
    # Configure 1 epoch for fast tests
    cfg = DefaultCFGKRURKConfig()
    cfg.cfkg.num_epochs = 1
    cfg.rurk.epochs = 1
    # CFKG needs num_blocks to unfreeze. The DummyModel has 2 blocks.
    cfg.cfkg.num_blocks = 1
    
    cfg = OmegaConf.structured(cfg)
    unlearner = CFGKRURKUnlearner(cfg, device="cpu")
    
    # Just verify it doesn't crash
    # It will automatically freeze during CFKG and unfreeze for RURK.
    with patch("munl.unlearning.catastrophic_forgetting_gamma_k.get_num_classes_from_model", return_value=2), \
         patch("munl.unlearning.catastrophic_forgetting_gamma_k.select_last_k_blocks", return_value=[model.conv]), \
         patch("munl.unlearning.catastrophic_forgetting_gamma_k.get_model_classifier", return_value=model.fc):
        updated_model = unlearner.unlearn(model, retain_loader, forget_loader, val_loader)
    
    assert updated_model is not None
