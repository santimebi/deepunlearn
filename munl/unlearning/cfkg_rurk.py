import typing as typ
from dataclasses import dataclass, field
from omegaconf import DictConfig
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from munl.unlearning.common import BaseUnlearner
from munl.unlearning.catastrophic_forgetting_gamma_k import ForgettingGammaK, DefaultForgettingGammaKConfig
from munl.unlearning.rurk import RURKUnlearner, DefaultRURKConfig
from munl.settings import default_loaders, DEFAULT_MODEL_INIT_DIR

@dataclass
class DefaultCFGKRURKConfig:
    cfkg: DefaultForgettingGammaKConfig = field(default_factory=DefaultForgettingGammaKConfig)
    rurk: DefaultRURKConfig = field(default_factory=DefaultRURKConfig)
    
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = None
    criterion: typ.Union[typ.Dict[str, typ.Any], None] = None
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)

class CFGKRURKUnlearner(BaseUnlearner):
    HYPER_PARAMETERS = {
        # Can be mapped to specific sub-configs if needed for optuna
    }

    def __init__(
        self,
        cfg: DictConfig,
        device,
        writer=None,
        save_steps: bool = False,
        should_evaluate: bool = False,
    ):
        super().__init__(
            cfg,
            device=device,
            writer=writer,
            save_steps=save_steps,
            should_evaluate=should_evaluate,
        )

    def unlearn(
        self,
        model: Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Module:
        print("[phase=cfkg] Starting Catastrophic Forgetting Gamma K")
        # Run CFKG
        cfkg_unlearner = ForgettingGammaK(
            self.cfg.cfkg, 
            device=self.device, 
            writer=self.writer, 
            save_steps=False, 
            should_evaluate=False
        )
        model = cfkg_unlearner.unlearn(model, retain_loader, forget_loader, val_loader)
        
        # Unfreeze entire model for RURK (since CFKG freezes non-last-k blocks)
        for param in model.parameters():
            param.requires_grad = True
            
        print("[phase=rurk] Starting RURK")
        # Run RURK
        rurk_unlearner = RURKUnlearner(
            self.cfg.rurk, 
            device=self.device, 
            writer=self.writer, 
            save_steps=False, 
            should_evaluate=False
        )
        model = rurk_unlearner.unlearn(model, retain_loader, forget_loader, val_loader)
        
        return model
