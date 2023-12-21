import pickle
import random
from pathlib import Path

import numpy as np
import torch
from graph_datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers.models.auto import AutoTokenizer


def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def compute_metrics(output, labels):
    preds = output.max(1)[1].type_as(labels)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    return {"w_f1": w_f1, "macro": macro, "micro": micro, "acc": acc}


def load_processed_dataset(dataset_name):
    file_path = Path(__file__).parent.parent.joinpath("data-processed", f"{dataset_name}.pkl")
    with open(file_path, "rb") as file:
        dataset: Dataset = pickle.load(file)
    return dataset


def get_encoder_outputs(dataset_name):
    emb_path = f"/Users/ardaaras/Documents/finetune-text-graphs/generator-output/{dataset_name}_embeddings.pth"
    cls_logits_path = f"/Users/ardaaras/Documents/finetune-text-graphs/generator-output/{dataset_name}_logits.pth"
    x = torch.load(emb_path)
    cls_logits = torch.load(cls_logits_path)
    return x, cls_logits


def get_encoder_input(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tmp = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=64)
    i_ids, a_mask = tmp["input_ids"], tmp["attention_mask"]
    return i_ids, a_mask


def get_loaders(dataset_name):
    dataset: Dataset = load_processed_dataset(dataset_name)
    i_ids, a_mask = get_encoder_input("roberta-base", list(dataset.doc_list))

    z = zip([dataset.train_ids, dataset.test_ids], ["train", "test"])
    datasets = {}
    loaders = {}
    for mask, split in z:
        datasets[split] = TensorDataset(i_ids[mask], a_mask[mask], dataset.y[mask], torch.tensor(mask))
        loaders[split] = DataLoader(datasets[split], batch_size=32, shuffle=True)

    return loaders


def get_A_s(dataset: Dataset, path):
    if path == "FF-NF":
        return [dataset.FF.to_dense().to("mps"), dataset.NF.to_dense().to("mps")]
    elif path == "FN-NF":
        return [dataset.FN.to_dense().to("mps"), dataset.NF.to_dense().to("mps")]
    elif path == "NN-NN":
        return [dataset.NN.to_dense().to("mps"), dataset.NN.to_dense().to("mps")]
    elif path == "NF-NN":
        return [dataset.NF.to_dense().to("mps"), dataset.NN.to_dense().to("mps")]
    else:
        raise ValueError("Path must be one of FF-NF, FN-NF, NN-NN, NF-NN")


def get_variables(model_type: str, path, dataset: Dataset):
    unit = torch.eye(1).to("mps")
    name = dataset.dataset_name
    if model_type == "type1":
        if path == "FF-NF":
            x = unit
            cls_logit = None
            fan_in = dataset.FF.shape[1]
            update_cls = False
        elif path == "FN-NF":
            x = unit
            cls_logit = None
            fan_in = dataset.FN.shape[1]
            update_cls = False
        elif path == "NF-NN":
            x = unit
            cls_logit = None
            fan_in = dataset.NF.shape[1]
            update_cls = False
        elif path == "NN-NN":
            x = unit
            cls_logit = None
            fan_in = dataset.NN.shape[1]
            update_cls = False
        else:
            raise ValueError("Path must be one of FF-NF, FN-NF, NN-NN, NF-NN")
    elif model_type == "type2":
        if path == "FF-NF":
            x = unit
            cls_logit = None
            fan_in = dataset.FF.shape[1]
            update_cls = False
        elif path == "FN-NF":
            x = get_encoder_outputs(name)[0]
            cls_logit = None
            fan_in = 768
            update_cls = False
        elif path == "NF-NN":
            x = unit
            cls_logit = None
            fan_in = dataset.NF.shape[1]
            update_cls = False
        elif path == "NN-NN":
            x = get_encoder_outputs(name)[0]
            cls_logit = None
            fan_in = 768
            update_cls = False
        else:
            raise ValueError("Path must be one of FF-NF, FN-NF, NN-NN, NF-NN")
    elif model_type == "type3":
        if path == "FF-NF":
            x = unit
            cls_logit = get_encoder_outputs(name)[1]
            fan_in = dataset.FF.shape[1]
            update_cls = False
        elif path == "FN-NF":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = 768
            update_cls = False
        elif path == "NF-NN":
            x = unit
            cls_logit = get_encoder_outputs(name)[1]
            fan_in = dataset.NF.shape[1]
            update_cls = False
        elif path == "NN-NN":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = 768
            update_cls = False
        else:
            raise ValueError("Path must be one of FF-NF, FN-NF, NN-NN, NF-NN")

    elif model_type == "type4":
        if path == "FF-NF":
            x = unit
            cls_logit = None
            fan_in = dataset.FF.shape[1]
            update_cls = False
        elif path == "FN-NF":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = 768
            update_cls = True
        elif path == "NF-NN":
            x = unit
            cls_logit = None
            fan_in = dataset.NF.shape[1]
            update_cls = False
        elif path == "NN-NN":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = 768
            update_cls = True
        else:
            raise ValueError("Path must be one of FF-NF, FN-NF, NN-NN, NF-NN")
    else:
        raise ValueError("Model type must be one of type1, type2, type3, type4")
    return x, cls_logit, fan_in, update_cls


def get_empty_results_dict():
    return {
        "dataset": [],
        "path": [],
        "type": [],
        "best_test_acc": [],
        "best_w_f1": [],
        "avg_epoch_time": [],
    }
