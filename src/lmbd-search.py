# %%
from pathlib import Path

import pandas as pd
import torch
from configs import SearchParams
from graph_datasets import Dataset  # noqa
from type_run import run_type

from src.util_fncs import get_empty_results_dict, set_seeds

# 20 NG DATASET
##ALL PATHS Type 1-2-3
params_dict = {
    "FF-NF": SearchParams(
        fan_mid=256,
        lmbd=0.65,
        gcn_lr=0.0007054,
        gcn_p=0.2142,
        wd=0.009779,
        max_epochs=1000,
        patience=10,
        encoder_lr=0,
        encoder_p=0,
    )
}

dataset_name = "20ng"
model_type = "type3"

set_seeds(seed_no=42)
results_d = get_empty_results_dict()
results_d["lmbd"] = []
for path in ["FF-NF", "FN-NF", "NN-NN", "NF-NN"]:
    for lmbd in torch.arange(0.0, 1.05, 0.05):
        params = params_dict["FF-NF"]
        params.lmbd = lmbd
        if lmbd == 0.0:
            params.patience = 0
        else:
            params.patience = 10
        trainer = run_type(dataset_name, model_type, path, params)
        results_d["dataset"].append(dataset_name)
        results_d["path"].append(path)
        results_d["type"].append("type3")
        results_d["best_test_acc"].append(f"{100*trainer.best_test_acc:.3f}")
        results_d["best_w_f1"].append(f"{100*trainer.best_w_f1:.3f}")
        results_d["avg_epoch_time"].append(f"{trainer.avg_epoch_time:.3f}")
        results_d["lmbd"].append(lmbd)
df = pd.DataFrame(results_d)
df["lmbd"] = df["lmbd"].astype(float)

# %%
FILE_DIR = Path(__file__).parent.parent.joinpath("plotting", "lmbd_sweep")
FILE_DIR.mkdir(exist_ok=True, parents=True)
file_name = f"{dataset_name}_{model_type}.csv"
file_path = FILE_DIR.joinpath(file_name)
df.to_csv(file_path, index=False)
