import copy
import time
from typing import Optional, Union

import torch
from configs import Type4Input, TypeInput
from models import Type3, Type4, Type12
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm.auto import tqdm
from utils import compute_metrics


class TypeTrainer:
    def __init__(self, model: Union[Type12, Type3], optimizer: Optimizer, t_input: TypeInput):
        self.model = model
        self.optimizer = optimizer
        self.input = t_input

    def pipeline(self, max_epochs: int, patience: int):
        es = EarlyStopping(patience=patience, verbose=True)
        t = tqdm(range(max_epochs))

        epoch_times, epoch_ct, best_w_f1 = [], 0, 0

        for epoch in t:
            start_time = time.time()
            self.model.train()
            e_loss = 0
            e_loss = self.train_epoch()
            metrics = self.evaluate()
            train_acc, test_acc = metrics["train"]["acc"], metrics["test"]["acc"]
            best_test_acc, best_model = es(test_acc, self.model, epoch)

            if metrics["test"]["w_f1"] > best_w_f1:
                best_w_f1 = metrics["test"]["w_f1"]
            t.set_description(f"Loss: {e_loss:.4f}, Best Test Acc: {100*best_test_acc:.3f}, Train Acc: {100*train_acc:.3f}")
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            epoch_ct += 1
            if es.early_stop:
                break
        self.avg_epoch_time = sum(epoch_times) / epoch_ct
        self.best_model = best_model
        self.best_test_acc = best_test_acc
        self.best_w_f1 = best_w_f1

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.input.x, self.input.A_s)
        loss = F.nll_loss(out[self.input.train_ids], self.input.y[self.input.train_ids])
        loss.backward()
        e_loss = loss.item()
        self.optimizer.step()
        return e_loss

    def evaluate(self):
        metrics = {}
        with torch.no_grad():
            self.model.eval()
            out = self.model(self.input.x, self.input.A_s)

            metrics["train"] = compute_metrics(out[self.input.train_ids], self.input.y[self.input.train_ids])
            metrics["test"] = compute_metrics(out[self.input.test_ids], self.input.y[self.input.test_ids])
        return metrics


class Type4Trainer:
    def __init__(self, model: Type4, optimizer: Optimizer, t_input: Type4Input):
        self.model = model
        self.optimizer = optimizer
        self.input = t_input

    def pipeline(self, max_epochs: int, patience: int, update_cls: bool = True):
        es = EarlyStopping(patience=patience, verbose=True)
        t = tqdm(range(max_epochs))

        epoch_times, epoch_ct, best_w_f1 = [], 0, 0

        for epoch in t:
            start_time = time.time()
            metrics = {}
            self.model.train()
            e_loss = 0
            metrics["test"] = self.evaluate(data_key="test")
            e_loss = self.train_epoch()
            metrics["test"] = self.evaluate(data_key="test")
            metrics["train"] = self.evaluate(data_key="train")

            train_acc, test_acc = metrics["train"]["acc"], metrics["test"]["acc"]
            best_test_acc, best_model = es(test_acc, self.model, epoch)
            if metrics["test"]["w_f1"] > best_w_f1:
                best_w_f1 = metrics["test"]["w_f1"]

            t.set_description(f"Loss: {e_loss:.4f}, Best Test Acc: {100*best_test_acc:.3f}, Train Acc: {100*train_acc:.3f}")

            if update_cls:
                self.update_cls()

            end_time = time.time()
            epoch_times.append(end_time - start_time)
            epoch_ct += 1

            if es.early_stop:
                break

        self.best_model = best_model
        self.best_test_acc = best_test_acc
        self.best_w_f1 = best_w_f1
        self.avg_epoch_time = sum(epoch_times) / epoch_ct

    def train_epoch(self):
        for i_ids, a_mask, y, idx in tqdm(self.input.loader["train"], leave=False):
            self.optimizer.zero_grad()
            i_ids, a_mask, y, idx = device_move("mps", i_ids, a_mask, y, idx)
            out = self.model(self.input.x, self.input.A_s, i_ids, a_mask, idx)

            loss = F.nll_loss(out, y)
            loss.backward()
            e_loss = loss.item()
            self.optimizer.step()
        return e_loss

    def update_cls(self):
        self.model.eval()
        with torch.no_grad():
            for i_ids, a_mask, _, idx in self.input.loader["train"]:
                out = self.model.encoder.transformer(i_ids.to("mps"), a_mask.to("mps"))[0][:, 0]
                self.input.x[idx] = out[idx]

            for i_ids, a_mask, _, idx in self.input.loader["test"]:
                out = self.model.encoder.transformer(i_ids.to("mps"), a_mask.to("mps"))[0][:, 0]
                self.input.x[idx] = out[idx]

    def evaluate(self, data_key="test"):
        with torch.no_grad():
            y_pred, y_true = [], []
            self.model.eval()
            for i_ids, a_mask, y, idx in self.input.loader[data_key]:
                i_ids, a_mask, y, idx = device_move("mps", i_ids, a_mask, y, idx)
                out = self.model(self.input.x, self.input.A_s, i_ids, a_mask, idx)

                y_pred.append(out.cpu())
                y_true.append(y.cpu())

            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            metrics = compute_metrics(y_pred, y_true)
        return metrics


def device_move(device, *tensors):
    moved_tensors = []
    for tensor in tensors:
        moved_tensor = tensor.to(device)
        moved_tensors.append(moved_tensor)
    return moved_tensors


class EarlyStopping:
    def __init__(self, patience: int, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose

        self.counter: int = 0
        self.best_test_acc: float = 0.0
        self.best_model: Optional[torch.nn.Module] = None
        self.early_stop = False

    def __call__(self, test_acc: float, model: torch.nn.Module, epoch: int):
        if test_acc > self.best_test_acc:
            self.counter = 0
            self.best_test_acc = test_acc
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping at epoch {epoch}")

        return self.best_test_acc, self.best_model
