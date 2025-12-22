import copy
from typing import List

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from thirdparty.repdistiller.helper.util import (
    adjust_learning_rate as sgda_adjust_learning_rate,
)

class CustomLoss(nn.Module):

    _BETA_STABILIZER = 1e-12 
    def __init__(self, model: Module, c: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.c = float(c)
        self.gamma = float(gamma)
        # Store the trainable parameters and their initial values (w0)
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._w0 = [p.detach().clone() for p in self._params]
        # Optional metrics for logging/debug
        self._latest_ce = 0.0
        self._latest_penalty = 0.0
        self._latest_norm = 0.0

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.cross_entropy(outputs, targets)

        dist_sq = None
        for p, p0 in zip(self._params, self._w0):
            d = p - p0
            term = (d * d).sum()
            dist_sq = term if dist_sq is None else (dist_sq + term)
        if dist_sq is None:
            norm = outputs.new_tensor(0.0)
        else:
            norm = torch.sqrt(dist_sq + self._BETA_STABILIZER)
        penalty = self.c * torch.exp(-(norm ** self.gamma))
        loss = loss_ce + penalty

        self._latest_ce = float(loss_ce.detach().cpu())
        self._latest_penalty = float(penalty.detach().cpu())
        self._latest_norm = float(norm.detach().cpu())
        return loss
    
def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(
        model.named_parameters(), model_init.named_parameters()
    ):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= weight_decay / 2.0
    return l2_loss


# NOTE: Adapted from https://github.com/meghdadk/SCRUB/blob/main/small_scale_unlearning.ipynb
def run_train_epoch(
    model: Module,
    model_init: Module,
    data_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    split: str,
    num_classes: int,
    weight_decay: float,
    delta_w=None,
    scrub_act=False,
):
    model.eval()
    assert split in ["train", "test"]

    with torch.set_grad_enabled(split != "test"):
        for _, batch in enumerate(tqdm(data_loader, leave=False)):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            input, target = batch
            output = model(input)
            if split == "test" and scrub_act:
                G = []
                for cls in range(num_classes):
                    grads = torch.autograd.grad(
                        output[0, cls], model.parameters(), retain_graph=True
                    )
                    grads = torch.cat([g.view(-1) for g in grads])
                    G.append(grads)

                G = torch.stack(G).pow(2)
                delta_f = torch.matmul(G, delta_w)
                output += delta_f.sqrt() * torch.empty_like(delta_f).normal_()
            loss = criterion(output, target) + l2_penalty(
                model, model_init, weight_decay
            )

            if split != "test":
                model.zero_grad()
                loss.backward()
                optimizer.step()
    return model


# Adapted from https://github.com/meghdadk/SCRUB/blob/main/small_scale_unlearning.ipynb
def fk_finetune(
    model: Module,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    num_classes: int,
    weight_decay: float,
    lr_decay_epochs: List[int],
    sgda_learning_rate: float,
    lr_decay_rate: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        sgda_adjust_learning_rate(
            epoch, lr_decay_epochs, sgda_learning_rate, lr_decay_rate, optimizer
        )
        run_train_epoch(
            model=model,
            model_init=model_init,
            data_loader=data_loader,
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            split="train",
            weight_decay=weight_decay,
        )

def fgk_finetune(
    model: Module,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    num_classes: int,
    weight_decay: float,
    lr_decay_epochs: List[int],
    sgda_learning_rate: float,
    lr_decay_rate: float,
    c: float = 1.0,
    gamma: float = 1.0,
):
    criterion = CustomLoss(model=model, c=c, gamma=gamma)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)

    for epoch in range(epochs):
        sgda_adjust_learning_rate(
            epoch, lr_decay_epochs, sgda_learning_rate, lr_decay_rate, optimizer
        )
        run_train_epoch(
            model=model,
            model_init=model_init,
            data_loader=data_loader,
            num_classes=num_classes,
            criterion=criterion,
            optimizer=optimizer,
            split="train",
            weight_decay=weight_decay,
        )
