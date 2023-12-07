from typing import List

import torch
import torch.nn as nn
from configs import EncoderConfig, Type3Config, Type4Config, Type12Config
from layers import GraphConvolution
from torch.nn import functional as F
from transformers import AutoModel


class BERTForSequenceClassification(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.c = config
        self.transformer = AutoModel.from_pretrained(config.model_name)
        fan_out = list(self.transformer.modules())[-2].out_features
        self.classifier = nn.Linear(fan_out, config.n_class)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.transformer(input_ids, attention_mask)[0][:, 0]
        out = self.dropout(cls_feats)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out


class Type12(nn.Module):
    def __init__(self, config: Type12Config):
        super().__init__()
        self.c = config
        self.gcn1 = GraphConvolution(self.c.fan_in, self.c.fan_mid)
        self.gcn2 = GraphConvolution(self.c.fan_mid, self.c.fan_out)

    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor]):
        x = self.gcn1(x, A_s[0])
        x = F.leaky_relu(x)
        x = F.dropout(x, self.c.dropout, training=self.training)
        x = self.gcn2(x, A_s[1])
        x = F.log_softmax(x, dim=1)
        return x


class Type3(nn.Module):
    def __init__(self, config: Type3Config):
        super().__init__()
        self.c = config
        self.gcn = Type12(self.c.type12_config)

    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor]):
        gcn_pred = self.gcn(x, A_s)  #! already log_softmax from model
        cls_pred = F.log_softmax(self.c.cls_logit, dim=1)

        pred = (gcn_pred) * self.c.lmbd + cls_pred * (1 - self.c.lmbd)
        return pred


class Type4(nn.Module):
    def __init__(self, config: Type4Config):
        super().__init__()
        self.c = config
        self.gcn = Type12(self.c.type12_config)
        self.encoder = BERTForSequenceClassification(self.c.encoder_config)

    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor], input_ids, attention_mask, idx):
        encoder_pred = self.encoder(input_ids, attention_mask)
        gcn_pred = self.gcn(x, A_s)[idx]
        pred = (gcn_pred) * self.c.lmbd + encoder_pred * (1 - self.c.lmbd)
        return pred
