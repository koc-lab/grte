from pathlib import Path
from typing import Optional

import yaml
from configs import SearchParams
from graph_datasets import Dataset  # noqa
from type_run import run_type

# import torch
import wandb
from src.util_fncs import set_seeds

file_path = Path(__file__).parent.joinpath("sweep_params.yaml")
with open(file_path) as f:
    sweep_params = yaml.load(f, Loader=yaml.FullLoader)  # noqa


model_type = sweep_params["name"]
print(f"Running sweep for {model_type}")


def run_sweep(c: Optional[dict] = None):
    run = wandb.init(config=c)  # noqa
    c = wandb.config
    if c is None:
        raise ValueError("You must specify the config dict for the sweep.")

    set_seeds(seed_no=42)
    search_params = SearchParams(
        fan_mid=c["fan_mid"],
        gcn_p=c["gcn_p"],
        lmbd=c["lmbd"],
        gcn_lr=c["gcn_lr"],
        wd=c["wd"],
        max_epochs=1000,
        patience=20,
    )
    trainer = run_type("20ng", model_type, c["path"], search_params)
    wandb.log(data={"test/best_test_acc": 100 * trainer.best_test_acc})
    wandb.log(data={"test/best_w_f1": 100 * trainer.best_w_f1})
    wandb.log(data={"avg_epoch_time": trainer.avg_epoch_time})


sweep_id = wandb.sweep(sweep_params, project="grte-20ng")
wandb.agent(sweep_id, project="grte-20ng", function=run_sweep)
