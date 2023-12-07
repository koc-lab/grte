from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class ModelParameters:
    model_type: str
    path: str
    fan_in: int
    n_class: int
    cls_logit: Optional[torch.Tensor] = None
    fan_mid: int = 200
    gcn_dropout: float = 0.0
    encoder_dropout: float = 0.0
    lmbd: float = 0.0
    encoder_ckpt: Optional[Path] = None


@dataclass
class EncoderConfig:
    model_name: str
    n_class: int
    dropout: float = 0.1


@dataclass
class Type12Config:
    fan_in: int
    fan_mid: int = 200
    fan_out: int = 2  # n_class
    dropout: float = 0.5


@dataclass
class Type3Config:
    type12_config: Type12Config
    cls_logit: torch.Tensor
    lmbd: float = 0.5


@dataclass
class Type4Config:
    type12_config: Type12Config
    encoder_config: EncoderConfig
    lmbd: float = 0.5


@dataclass
class TypeInput:
    x: torch.Tensor
    A_s: torch.Tensor
    y: torch.Tensor
    train_ids: torch.Tensor
    test_ids: torch.Tensor


@dataclass
class Type4Input:
    x: torch.Tensor
    A_s: torch.Tensor
    loader: DataLoader


@dataclass
class SearchParams:
    fan_mid: int
    gcn_p: float  # dropout prob
    encoder_p: float
    lmbd: float
    gcn_lr: float
    encoder_lr: float
    wd: float
    max_epochs: int
    patience: int
