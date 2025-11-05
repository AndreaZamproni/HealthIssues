from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset




def load_dataframe(csv_path: Path) -> pd.DataFrame:
if not csv_path.exists():
    raise FileNotFoundError(f"CSV non trovato: {csv_path}")
return pd.read_csv(csv_path)




def train_val_test_numpy(df: pd.DataFrame, target_col: str, test_size: float, seed: int):
   from sklearn.model_selection import train_test_split


y = df[target_col].values.astype(np.int64)
X = df.drop(columns=[target_col]).astype(np.float32).values


X_trv, X_te, y_trv, y_te = train_test_split(
X, y, test_size=test_size, random_state=seed, stratify=y
)
X_tr, X_va, y_tr, y_va = train_test_split(
X_trv, y_trv, test_size=test_size, random_state=seed, stratify=y_trv
)


# normalizzazione min-max sulla train
tr_min = X_tr.min(axis=0)
tr_max = X_tr.max(axis=0)
eps = 1e-8
norm = lambda Z: (Z - tr_min) / (tr_max - tr_min + eps)


return norm(X_tr), norm(X_va), norm(X_te), y_tr, y_va, y_te




def to_datasets(X_tr, y_tr, X_va, y_va, X_te, y_te):
tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
te = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
return tr, va, te