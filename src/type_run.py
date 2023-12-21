from pathlib import Path

import torch
from configs import EncoderConfig, SearchParams, Type3Config, Type4Config, Type12Config
from models import Type3, Type4, Type12
from trainers import Type4Input, Type4Trainer, TypeInput, TypeTrainer

from src.util_fncs import get_A_s, get_loaders, get_variables, load_processed_dataset


def run_type12(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, _, fan_in, _ = get_variables(model_type, path, data)
    config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    model = Type12(config).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y, data.train_ids, data.test_ids)
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)
    return trainer


def run_type3(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, cls_logit, fan_in, _ = get_variables(model_type, path, data)
    gcn_config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    config = Type3Config(gcn_config, cls_logit, lmbd=params.lmbd)
    model = Type3(config).to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y, data.train_ids, data.test_ids)
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)
    return trainer


def run_type4(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    loaders = get_loaders(dataset_name=dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    x, _, fan_in, update_cls = get_variables(model_type, path, data)
    gcn_config = Type12Config(fan_in, params.fan_mid, n_class, params.gcn_p)
    encoder_config = EncoderConfig("roberta-base", n_class, params.encoder_p)
    config = Type4Config(gcn_config, encoder_config, lmbd=params.lmbd)
    model = Type4(config).to("mps")
    enc_ckpt = Path(__file__).parent.parent.joinpath("encoder_ckpts", f"{dataset_name}.pth")
    print(f"Loading encoder from {enc_ckpt}")
    model.encoder.load_state_dict(torch.load(enc_ckpt))
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


def run_type(dataset_name: str, model_type: str, path: str, params: SearchParams):
    if model_type == "type1" or model_type == "type2":
        return run_type12(dataset_name, model_type, path, params)
    elif model_type == "type3":
        return run_type3(dataset_name, model_type, path, params)
    elif model_type == "type4":
        return run_type4(dataset_name, model_type, path, params)
    else:
        raise ValueError(f"Unknown model type {model_type}")


# %%
