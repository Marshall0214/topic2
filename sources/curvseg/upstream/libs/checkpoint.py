import os
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
    model_name: str = "checkpoint",
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, f"{model_name}.pth"))


def resume(
    result_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    model_name: str = "checkpoint",
) -> Tuple[Any]:
    resume_path = result_path
    if resume_path.endswith(".pth") is False:
        # resume_path = os.path.join(result_path, model_name)
        resume_path = os.path.join(result_path, f"{model_name}.pth")
    print("loading checkpoint {}".format(resume_path))

    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    # confirm whether the optimizer matches that of checkpoints
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError as e:
        print(f"[Warning] optimizer state mismatch, skip loading optimizer state:\n  {e}")

    return begin_epoch, model, optimizer, best_loss
