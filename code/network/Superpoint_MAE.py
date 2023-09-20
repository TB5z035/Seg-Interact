import numpy as np
import torch
import torch.nn as nn

from . import register_network


class PointNet():

    def __init__(self):
        raise NotImplementedError


class MAE_Encoder():

    def __init__(self):
        raise NotImplementedError


class MAE_Decoder():

    def __init__(self):
        raise NotImplementedError


class PC_Projector():

    def __init__(self):
        raise NotImplementedError


@register_network('Superpoint_MAE')
class Superpoint_MAE():

    def __init__(self):
        raise NotImplementedError


'''Stage 2'''


class MPN_MLP(nn.module):

    def __init__(self, mlp_in_dim=8):
        super().__init__()
        self.mlp_in_dim = mlp_in_dim
        self.fc_layer = nn.Linear(self.in_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, encoder_features):
        logits = self.fc_layer(encoder_features)
        print(logits.shape)
        scores = self.softmax(torch.squeeze(logits, dim=1))
        print(scores, scores.shape)
        return scores


class MPN_Encoder(nn.module):

    def __init__(self, encoder_in_dim=10, encoder_out_dim=8):
        super().__init__()
        self.encoder_in_dim = encoder_in_dim
        self.encoder_out_dim = encoder_out_dim
        self.qkv = nn.Linear(self.encoder_in_dim, self.encoder_out_dim)

    def forward(self, point_features):
        return encoder_features


@register_network('MPN')
class MPN(MPN_Encoder, MPN_MLP):

    def __init__(self, dims):
        self.encoder = MPN_Encoder()
        self.mlp = MPN_MLP()

    def forward(self):
        return mask_sp_indices
