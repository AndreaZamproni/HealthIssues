import os
import random
from pathlib import Path
import numpy as np
import torch




def seed_everything(seed: int) -> None:
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True




def get_device() -> torch.device:
return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def ensure_dirs(*dirs: Path) -> None:
for d in dirs:
d.mkdir(parents=True, exist_ok=True)




def make_loader(ds, batch_size, shuffle, drop_last):
cpu_cores = os.cpu_count() or 2
num_workers = max(2, min(4, cpu_cores))
pin = torch.cuda.is_available()
return torch.utils.data.DataLoader(
ds,
batch_size=batch_size,
shuffle=shuffle,
drop_last=drop_last,
num_workers=num_workers,
pin_memory=pin,
pin_memory_device="cuda" if pin else "",
prefetch_factor=4,
)