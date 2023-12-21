# %%
import pandas as pd
from configs import SearchParams
from graph_datasets import Dataset  # noqa
from type_run import run_type

from src.util_fncs import get_empty_results_dict, set_seeds

params = SearchParams(
    fan_mid=256,
    lmbd=0.65,
    gcn_lr=0.0007054,
    gcn_p=0.3,
    wd=0.009779,
    patience=3,
    max_epochs=10,
)
# above gives 89.613 for FF-NF on 20ng
trainer = run_type(dataset_name="20ng", model_type="type1", path="FF-NF", params=params)
# %%

d = get_empty_results_dict()
set_seeds(seed_no=1)
for name in ["mr", "R8", "R52", "ohsumed", "20ng"]:
    for path in ["FF-NF", "FN-NF", "NN-NN", "NF-NN"]:
        trainer = run_type(dataset_name=name, model_type="type1", path=path, params=params)
    d["dataset"].append(name)
    d["path"].append(path)
    d["type"].append("type4")
    d["best_test_acc"].append(trainer.best_test_acc)
    d["avg_epoch_time"].append(trainer.avg_epoch_time)

# %%


df = pd.DataFrame(d)
