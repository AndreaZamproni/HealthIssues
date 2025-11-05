import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from src.config import DATA_DIR, CSV_NAME
from src.utils import seed_everything, get_device, ensure_dirs, make_loader
from src.data import load_dataframe
from src.model import FeedForwardNet




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_DIR / CSV_NAME)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--hidden_layers", type=int, required=True)
    p.add_argument("--hidden_size", type=int, required=True)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", type=Path, required=True, help="cartella con split_i_model.pt")
    args = p.parse_args()


    seed_everything(args.seed)
    device = get_device()


    df = load_dataframe(args.data)
    X = df.drop(columns=["Health_Issues"]).astype("float32").values
    y = df["Health_Issues"].astype("int64").values


    test_acc, test_prec, test_rec, test_f1 = [], [], [], []
    all_tgts, all_preds = [], []


    for split in range(args.k):
        from sklearn.model_selection import train_test_split
        X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed+split, stratify=y)
        X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=args.test_size, random_state=args.seed+split, stratify=y_trv)
        tr_min, tr_max = X_tr.min(0), X_tr.max(0)
        eps = 1e-8
        norm = lambda Z: (Z - tr_min) / (tr_max - tr_min + eps)
        X_te_n = norm(X_te)


    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_te_n), torch.from_numpy(y_te))
    te_loader = make_loader(te_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)


    model = FeedForwardNet(in_features=X.shape[1], hidden_layers=args.hidden_layers, hidden_size=args.hidden_size, dropout_rate=args.dropout, num_classes=len(np.unique(y))).to(device)
    ckpt = args.ckpt_dir / f"split_{split}_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()


    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            tgts.append(yb.numpy())
            preds = np.concatenate(preds); tgts = np.concatenate(tgts)


    test_acc.append(accuracy_score(tgts, preds))
    test_prec.append(precision_score(tgts, preds, average='weighted'))
    test_rec.append(recall_score(tgts, preds, average='weighted'))
    test_f1.append(f1_score(tgts, preds, average='weighted'))
    all_tgts.extend(tgts); all_preds.extend(preds)
    print(f"Split {split+1}: F1={test_f1[-1]:.4f}")


    print("\nAverages on test:")
    print(f"Accuracy: {np.mean(test_acc):.4f} ± {np.std(test_acc):.4f}")
    print(f"Precision: {np.mean(test_prec):.4f} ± {np.std(test_prec):.4f}")
    print(f"Recall: {np.mean(test_rec):.4f} ± {np.std(test_rec):.4f}")
    print(f"F1: {np.mean(test_f1):.4f} ± {np.std(test_f1):.4f}")


    cm = confusion_matrix(all_tgts, all_preds)
    print("Confusion matrix:\n", cm)




    if __name__ == "__main__":
        main()