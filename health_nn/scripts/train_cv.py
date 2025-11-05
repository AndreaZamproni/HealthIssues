import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler


from src.config import ROOT, DATA_DIR, MODELS_DIR, RUNS_DIR, CSV_NAME, TrainCfg, ArchCfg, OptCfg
from src.utils import seed_everything, ensure_dirs, get_device, make_loader
from src.data import load_dataframe, train_val_test_numpy, to_datasets
from src.model import FeedForwardNet
from src.training import fit
from src.cv import k_shuffle_split_cv




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_DIR / CSV_NAME)
    p.add_argument("--epochs", type=int, default=TrainCfg.epochs)
    p.add_argument("--patience", type=int, default=TrainCfg.patience)
    p.add_argument("--k", type=int, default=TrainCfg.k_splits)
    p.add_argument("--test_size", type=float, default=TrainCfg.test_size)
    p.add_argument("--hidden_layers", type=int, default=ArchCfg.hidden_layers)
    p.add_argument("--hidden_size", type=int, default=ArchCfg.hidden_size)
    p.add_argument("--dropout", type=float, default=ArchCfg.dropout_rate)
    p.add_argument("--batch_size", type=int, default=OptCfg.batch_size)
    p.add_argument("--lr", type=float, default=OptCfg.learning_rate)
    p.add_argument("--l1", type=float, default=OptCfg.l1_lambda)
    p.add_argument("--l2", type=float, default=OptCfg.l2_lambda)
    p.add_argument("--seed", type=int, default=TrainCfg.seed)
    p.add_argument("--out", type=Path, default=MODELS_DIR / "baseline")
    args = p.parse_args()


    ensure_dirs(args.out, RUNS_DIR, MODELS_DIR)
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")


    df = load_dataframe(args.data)
    X = df.drop(columns=["Health_Issues"]).astype("float32").values
    y = df["Health_Issues"].astype("int64").values


    scaler = GradScaler(enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss()


    losses, metrics, scores = k_shuffle_split_cv(
        X=X, y=y,
        epochs=args.epochs, criterion=crit, scaler=scaler, device=device,
        k=args.k, test_size=args.test_size, batch_size=args.batch_size,
        hidden_layers=args.hidden_layers, hidden_size=args.hidden_size,
        learning_rate=args.lr, dropout_rate=args.dropout,
        l1_lambda=args.l1, l2_lambda=args.l2, patience=args.patience,
        verbose=10, seed=args.seed, out_dir=args.out
    )
    print(scores)




if __name__ == "__main__":
    main()