from itertools import product
from pathlib import Path
import numpy as np
from .cv import k_shuffle_split_cv




def grid_search_cv(X, y, param_grid: dict, fixed_params: dict, cv_params: dict, out_root: Path | None = None, verbose=True):
    names, values = list(param_grid.keys()), list(param_grid.values())
    combos = list(product(*values))
    results = {}
    best_score = -np.inf
    best_config = None


    for i, combo in enumerate(combos, 1):
        cur = dict(zip(names, combo))
        cfg_name = "_".join([f"{k}_{v}" for k, v in cur.items()])
        if verbose:
            print(f"\nConfiguration {i}/{len(combos)}: {cur}")
        run_params = {**fixed_params, **cur}
        out_dir = (out_root / cfg_name) if out_root else None
        _, _, fold_scores = k_shuffle_split_cv(X, y, out_dir=out_dir, **run_params, **cv_params)
        results[cfg_name] = fold_scores
        if fold_scores["mean"] > best_score:
            best_score = fold_scores["mean"]
            best_config = cur.copy()
            if verbose:
                print(" NEW BEST SCORE")
        if verbose:
            print(f" F1 Score: {fold_scores['mean']:.4f}±{fold_scores['std']:.4f}")
    return results, best_config, best_score