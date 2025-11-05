import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


from .model import FeedForwardNet
from .training import fit
from .utils import make_loader




def k_shuffle_split_cv(X, y, *, epochs, criterion, scaler, device,
k, test_size, batch_size, hidden_layers, hidden_size, learning_rate, dropout_rate,
l1_lambda=0.0, l2_lambda=0.0, patience=0, verbose=10, seed=42, out_dir: Path | None = None):

    fold_losses, fold_metrics, best_scores = {}, {}, {}


    in_features = X.shape[1]
    num_classes = len(np.unique(y))
    model = FeedForwardNet(in_features, hidden_layers, hidden_size, dropout_rate, num_classes).to(device)
    init_state = copy.deepcopy(model.state_dict())


    for split_idx in range(k):
        if verbose:
            print(f"Split {split_idx+1}/{k}")
        X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=test_size, random_state=seed+split_idx, stratify=y)
        X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=test_size, random_state=seed+split_idx, stratify=y_trv)


        tr_max = X_tr.max(axis=0); tr_min = X_tr.min(axis=0); eps = 1e-8
        norm = lambda Z: (Z - tr_min) / (tr_max - tr_min + eps)
        X_tr, X_va, X_te = norm(X_tr), norm(X_va), norm(X_te)


        tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        va_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        te_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))


        tr_loader = make_loader(tr_ds, batch_size, True, False)
        va_loader = make_loader(va_ds, batch_size, False, False)
        _ = make_loader(te_ds, batch_size, False, False) # non usato qui


        model.load_state_dict(init_state)
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)


        ckpt = None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = out_dir / f"split_{split_idx}_model.pt"
        writer = SummaryWriter(log_dir=str(out_dir / f"tb_split_{split_idx}") if out_dir else None)


        model, hist = fit(
            model=model, train_loader=tr_loader, val_loader=va_loader, epochs=epochs,
            criterion=criterion, optimizer=optim, scaler=scaler, device=device,
            l1_lambda=l1_lambda, l2_lambda=l2_lambda, patience=patience,
            writer=writer, verbose=verbose, ckpt_path=str(ckpt) if ckpt else None
        )


        fold_losses[f"split_{split_idx}"] = hist["val_loss"]
        fold_metrics[f"split_{split_idx}"] = hist["val_f1"]
        best_scores[f"split_{split_idx}"] = float(np.max(hist["val_f1"]))


    best_scores["mean"] = float(np.mean([best_scores[k] for k in best_scores.keys()]))
    best_scores["std"] = float(np.std([best_scores[k] for k in best_scores.keys()]))
    print(f"Best score: {best_scores['mean']:.4f}±{best_scores['std']:.4f}")
    return fold_losses, fold_metrics, best_scores