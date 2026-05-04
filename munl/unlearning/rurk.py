from dataclasses import dataclass, field
import itertools
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import typing as typ

import munl.settings
from munl.models import get_optimizer_scheduler_criterion
from munl.unlearning.common import BaseUnlearner
from munl.evaluation.residual_knowledge import gaussian_perturb
from munl.settings import default_loaders, DEFAULT_MODEL_INIT_DIR, default_criterion, default_scheduler

def rurk_default_optimizer():
    return {
        "type": "torch.optim.SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    }

@dataclass
class DefaultRURKConfig:
    epochs: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lambda_f: float = 0.03
    lambda_a: float = 0.00045
    tau: float = 0.03
    num_adv_samples: int = 1
    batch_size: int = 128
    
    optimizer: typ.Dict[str, typ.Any] = field(
        default_factory=rurk_default_optimizer
    )
    scheduler: typ.Union[typ.Dict[str, typ.Any], None] = field(
        default_factory=default_scheduler
    )
    criterion: typ.Dict[str, typ.Any] = field(default_factory=default_criterion)
    model_initializations_dir: str = DEFAULT_MODEL_INIT_DIR
    loaders: typ.Dict[str, typ.Any] = field(default_factory=default_loaders)


class RURKUnlearner(BaseUnlearner):
    HYPER_PARAMETERS = munl.settings.HYPER_PARAMETERS

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
        model: nn.Module,
        retain_loader: DataLoader,
        forget_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        device = self.device
        
        # Override optimizer arguments based on config
        # This matches the signature of typical unlearners
        from omegaconf import OmegaConf
        OmegaConf.set_struct(self.cfg, False)
        self.cfg.num_epochs = self.cfg.epochs
        OmegaConf.set_struct(self.cfg, True)
        optimizer, scheduler, criterion = get_optimizer_scheduler_criterion(model, self.cfg)
        model.to(device)
        model.train()

        generator = torch.Generator(device=device)

        # We cycle forget loader in case it's shorter than retain loader
        forget_iterator_cycle = itertools.cycle(forget_loader)

        for epoch in tqdm(range(self.cfg.epochs), desc="RURK Epochs"):
            epoch_retain_loss = 0.0
            epoch_forget_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_total_loss = 0.0
            batches = 0
            
            for x_r, y_r in retain_loader:
                x_r, y_r = x_r.to(device), y_r.to(device)
                
                # Get next forget batch
                x_f, y_f = next(forget_iterator_cycle)
                x_f, y_f = x_f.to(device), y_f.to(device)

                optimizer.zero_grad()

                # Retain Loss
                logits_r = model(x_r)
                retain_loss = criterion(logits_r, y_r)

                # Forget Loss
                logits_f = model(x_f)
                forget_loss = criterion(logits_f, y_f)

                # Adversarial Forget Loss
                x_adv = gaussian_perturb(x_f, tau=self.cfg.tau, generator=generator)
                logits_adv = model(x_adv)
                adv_forget_loss = criterion(logits_adv, y_f)

                # Total Loss
                loss = retain_loss - self.cfg.lambda_f * forget_loss - self.cfg.lambda_a * adv_forget_loss
                loss.backward()
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_retain_loss += retain_loss.item()
                epoch_forget_loss += forget_loss.item()
                epoch_adv_loss += adv_forget_loss.item()
                epoch_total_loss += loss.item()
                batches += 1

            val_batch_loss = self.evaluate_if_needed(
                model, val_loader, criterion, device
            )
            
            payload = {
                "train_loss": epoch_total_loss / max(1, batches),
                "retain_loss": epoch_retain_loss / max(1, batches),
                "forget_loss": epoch_forget_loss / max(1, batches),
                "adv_forget_loss": epoch_adv_loss / max(1, batches),
                "val_loss": val_batch_loss.mean() if val_batch_loss is not None else 0.0,
            }
            self.save_and_log(model, optimizer, scheduler, payload, epoch)
            
        return model
