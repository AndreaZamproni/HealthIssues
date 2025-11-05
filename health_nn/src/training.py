rom typing import Tuple, Dict, List
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


@torch.no_grad()
def _predict_logits(model, xb, device):
    xb = xb.to(device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        return model(xb)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda=0.0, l2_lambda=0.0) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_pred, all_tgt = [], []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1_lambda:
                l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1
            if l2_lambda:
                l2 = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)
        all_pred.append(logits.argmax(1).detach().cpu().numpy())
        all_tgt.append(yb.detach().cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = f1_score(np.concatenate(all_tgt), np.concatenate(all_pred), average="weighted")
    return epoch_loss, epoch_f1


def validate_one_epoch(model, val_loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_tgt.append(yb.cpu().numpy())
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_f1 = f1_score(np.concatenate(all_tgt), np.concatenate(all_pred), average="weighted")
    return epoch_loss, epoch_f1


def log_tb(writer: SummaryWriter, epoch: int, train_loss: float, train_f1: float, val_loss: float, val_f1: float, model: torch.nn.Module):
    writer.add_scalar("Loss/Training", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("F1/Training", train_f1, epoch)
    writer.add_scalar("F1/Validation", val_f1, epoch)
    for name, p in model.named_parameters():
        if p.requires_grad and p.numel() > 0:
            writer.add_histogram(f"{name}/weights", p.data, epoch)
            if p.grad is not None and p.grad.numel() > 0:
                writer.add_histogram(f"{name}/grads", p.grad.data, epoch)


def fit(model, train_loader, val_loader, epochs, criterion, optimizer, scaler, device,
l1_lambda=0.0, l2_lambda=0.0, patience=0, evaluation_metric="val_f1", mode="max",
restore_best_weights=True, writer: SummaryWriter | None = None, verbose=10, ckpt_path=None):

    hist: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    best_metric = float("-inf") if mode == "max" else float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda, l2_lambda)
        va_loss, va_f1 = validate_one_epoch(model, val_loader, criterion, device)
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["train_f1"].append(tr_f1); hist["val_f1"].append(va_f1)
        if writer:
            log_tb(writer, epoch, tr_loss, tr_f1, va_loss, va_f1, model)
        if verbose and (epoch == 1 or epoch % verbose == 0):
            print(f"Epoch {epoch:3d}/{epochs} | Train: L={tr_loss:.4f}, F1={tr_f1:.4f} | Val: L={va_loss:.4f}, F1={va_f1:.4f}")

        current = va_f1 if evaluation_metric == "val_f1" else va_loss
        improved = current > best_metric if mode == "max" else current < best_metric
        if improved:
            best_metric = current
            best_epoch = epoch
            patience_counter = 0
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if restore_best_weights and ckpt_path and Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Best model restored from epoch {best_epoch}")

    if writer:
        writer.close()
    return model, hist