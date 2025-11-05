from pathlib import Path
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"


@dataclass
class TrainCfg:
seed: int = 42
epochs: int = 100
patience: int = 20
verbose: int = 10
test_size: float = 0.2
k_splits: int = 5


@dataclass
class ArchCfg:
hidden_layers: int = 5
hidden_size: int = 128
dropout_rate: float = 0.0


@dataclass
class OptCfg:
batch_size: int = 64
learning_rate: float = 3e-3
l1_lambda: float = 0.0
l2_lambda: float = 0.0


CSV_NAME = "dataset_numeric.csv"