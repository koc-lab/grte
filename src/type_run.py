from pathlib import Path

import torch
from configs import EncoderConfig, SearchParams, Type3Config, Type4Config, Type12Config
from models import Type3, Type4, Type12
from trainers import Type4Input, Type4Trainer, TypeInput, TypeTrainer
from utils import foo_dataset, get_A_s, get_loaders, get_variables


def run_type12(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = foo_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, _, fan_in, _ = get_variables(model_type, path, data)
    config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    model = Type12(config).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y, data.train_ids, data.test_ids)
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)
    return trainer


def run_type3(dataset_name: str, path: str, params: SearchParams):
    data = foo_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, cls_logit, fan_in, _ = get_variables("type3", path, data)
    gcn_config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    config = Type3Config(gcn_config, cls_logit, lmbd=params.lmbd)
    model = Type3(config).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y, data.train_ids, data.test_ids)
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)
    return trainer


def run_type4(dataset_name: str, path: str, params: SearchParams):
    data = foo_dataset(dataset_name)
    loaders = get_loaders(dataset_name=dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, _, fan_in, update_cls = get_variables("type4", path, data)
    gcn_config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    encoder_config = EncoderConfig("roberta-base", n_class, params.encoder_p)
    config = Type4Config(gcn_config, encoder_config, lmbd=params.lmbd)
    model = Type4(config).to("mps")
    enc_ckpt = Path(__file__).parent.joinpath("encoder_ckpts", f"{dataset_name}.pth")
    model.encoder.load_state_dict(enc_ckpt)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": params.encoder_lr},
            {"params": model.gcn.parameters(), "lr": params.gcn_lr},
        ],
        weight_decay=params.wd,
    )
    t_input = Type4Input(x=x, A_s=get_A_s(data, path), loader=loaders)
    trainer = Type4Trainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience, update_cls=update_cls)
    return trainer


# results_d = get_empty_results_dict()
# for path in ["FF-NF", "FN-NF", "NN-NN", "NF-NN"]:
#     for model_type in ["type1", "type2", "type3"]:
#         set_seeds(seed_no=42)
#         dataset: Dataset = foo_dataset(dataset_name="20ng")
#         x, cls_logit, fan_in, update_cls = get_variables(model_type, path, dataset)
#         type12_config = Type12Config(fan_in, 200, 20, dropout=0.3)

#         if model_type == "type1" or model_type == "type2":
#             model = Type12(type12_config).to("mps")

#         elif model_type == "type3":
#             config = Type3Config(type12_config, cls_logit, lmbd=0.3)
#             model = Type3(config).to("mps")

#         t_input = TypeInput(
#             x, get_A_s(dataset, path), dataset.y, dataset.train_ids, dataset.test_ids
#         )
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
#         trainer = TypeTrainer(model, optimizer, t_input)
#         trainer.pipeline(1000, 50)

#         results_d["dataset"].append(dataset.dataset_name)
#         results_d["path"].append(path)
#         results_d["type"].append(model_type)
#         results_d["best_test_acc"].append(f"{100*trainer.best_test_acc:.3f}")
#         results_d["best_w_f1"].append(f"{100*trainer.best_w_f1:.3f}")
#         results_d["avg_epoch_time"].append(f"{trainer.avg_epoch_time:.3f}")
#         DATASET_DIR = Path("/Users/ardaaras/Documents/grte/results/20ng")
#         DATASET_DIR.mkdir(exist_ok=True, parents=True)
#         TYPE_DIR = DATASET_DIR.joinpath(f"{model_type}")
#         TYPE_DIR.mkdir(exist_ok=True, parents=True)
#         file_name = f"{path}_{100*trainer.best_test_acc:.3f}.pth"
#         torch.save(trainer.best_model.state_dict(), TYPE_DIR.joinpath(file_name))
# pd.DataFrame(results_d)
# %%
