import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import numpy as np
from typing import Optional
import copy
from . import register_network
from ..sp_dependencies.chamfer3D import dist_chamfer_3D


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0., inplace: bool = False):

        super().__init__()
        self.drop_prob = drop_prob
        self.inplace = inplace

    def drop_path(self, x: torch.Tensor, drop_prob: float = 1.0, inplace: bool = False) -> torch.Tensor:
        mask_shape: tuple[int] = (x.shape[0],) + (1,) * (x.ndim - 1)
        # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
        mask: torch.Tensor = x.new_empty(mask_shape).bernoulli_(drop_prob)
        mask.div_(drop_prob)
        if inplace:
            x.mul_(mask)
        else:
            x = x * mask
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            x = self.drop_path(x, self.drop_prob, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.keep_prob})"


class Token_Embed(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 token_embed_dim: int,
                 hidden_dim_1: int = 128,
                 hidden_dim_2: int = 256,
                 hidden_dim_3: int = 512):

        super().__init__()
        self.first_conv = nn.Sequential(nn.Conv1d(feature_dim, hidden_dim_1, 1),
                                        nn.Conv1d(hidden_dim_1, hidden_dim_2, 1))
        self.second_conv = nn.Sequential(nn.Conv1d(hidden_dim_3, hidden_dim_3, 1), nn.ReLU(inplace=True),
                                         nn.Conv1d(hidden_dim_3, token_embed_dim, 1))

    def mini_pointnet(self, point_groups):
        '''
            point_groups : N 11
            -----------------
            feature_global : C
        '''

        n, _ = point_groups.shape
        # encoder
        feature = self.first_conv(point_groups.transpose(0, 1))  # 256 n
        feature_global = torch.max(feature, dim=1, keepdim=True)[0]  # 256 1
        feature = torch.cat([feature_global.expand(-1, n), feature], dim=0)  # BG 512 n
        feature = self.second_conv(feature)  # 1024 n
        feature_global = torch.max(feature, dim=1, keepdim=False)[0]  # 1024
        feature_global = torch.unsqueeze(feature_global, dim=0)
        return feature_global

    def forward(self, full_features, full_super_indices):
        sp_features_batch = []
        for i, full_feature in enumerate(full_features):
            sp_max = len(torch.unique(full_super_indices[i]))
            sp_features = []
            for sp_index in range(sp_max):
                point_features = full_feature[torch.where(full_super_indices[i] == sp_index)[0]].to(
                    torch.float32).cuda()
                sp_feature = self.mini_pointnet(point_features)
                sp_features.append(sp_feature)
            sp_features = torch.cat(sp_features, dim=0)
            sp_features_batch.append(sp_features)
        return sp_features_batch


class Pos_Embed(nn.Module):

    def __init__(self, output_dim: int, pos_embed_hidden_dim: int = 128):

        super().__init__()

        self.pos_embed = nn.Sequential(nn.Linear(3, pos_embed_hidden_dim), nn.GELU(),
                                       nn.Linear(pos_embed_hidden_dim, output_dim))

    def forward(self, batch_coords):
        batch_pos_embed = []
        for i in range(len(batch_coords)):
            batch_pos_embed.append(self.pos_embed(batch_coords[i].cuda()))
        return batch_pos_embed


class Embed_and_Prep(nn.Module):

    def __init__(self, sp_feature_dim: int, sp_embed_dim: int, pad_limit: int = 64, mask_ratio: float = 0.6):

        super().__init__()
        self.pad_limit = pad_limit
        self.mask_ratio = mask_ratio
        self.batch_sp2_set = None
        self.batch_convert_remain_indices = []
        self.batch_convert_mask_indices = []

        self.token_embed = Token_Embed(feature_dim=sp_feature_dim, token_embed_dim=sp_embed_dim)
        self.pos_embed = Pos_Embed(output_dim=sp_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def mask_token_embed(self, batch_token_embed):
        batch_mask_embed = []
        batch_remain_embed = []
        batch_mask_indices = []
        batch_remain_indices = []

        for batch_index in range(len(batch_token_embed)):
            N, C = batch_token_embed[batch_index].shape
            mask_num = int(N * self.mask_ratio)
            mask_indices = torch.from_numpy(np.sort(np.random.choice(N, mask_num, replace=False)))
            remain_indices = torch.tensor([num for num in range(N) if num not in mask_indices])
            batch_mask_embed.append(batch_token_embed[batch_index][mask_indices])
            batch_remain_embed.append(batch_token_embed[batch_index][remain_indices])
            batch_mask_indices.append(mask_indices)
            batch_remain_indices.append(remain_indices)

        return batch_mask_embed, batch_remain_embed, batch_mask_indices, batch_remain_indices

    def pad_embed(self, batch_token_embed, higher_full_super_indices, batch_indices=None, record_mode=''):
        assert record_mode in ['', 'remain', 'mask'], 'check pad_embed record_mode'
        batch_tokens = []
        self.batch_sp2_set = []

        for i in range(len(higher_full_super_indices)):
            sp2_set = torch.unique(higher_full_super_indices[i])
            self.batch_sp2_set.append(sp2_set)
            tokens = []
            scene_sp2_2_sp1_token_indices = []

            for sp2_index in sp2_set:
                if batch_indices != None:
                    sp2_to_sp_indices = torch.where(higher_full_super_indices[i] == sp2_index)[0]
                    for j in range(len(batch_indices)):
                        batch_indices[j] = batch_indices[j].cuda()
                    sp2_to_sp_token_indices = torch.tensor(sorted(
                        [index for index in sp2_to_sp_indices if index in batch_indices[i]]),
                                                           dtype=int)
                    sp2_token = batch_token_embed[i][sp2_to_sp_token_indices]
                    assert sp2_token.dim(
                    ) == 2, f'sp2_remain_token has dim: {sp2_token.dim()} which is invalid for padding'
                else:
                    sp2_token = batch_token_embed[i]
                pad_dim = self.pad_limit - sp2_token.shape[0]
                sp2_token = F.pad(sp2_token, (0, 0, 0, pad_dim))
                tokens.append(sp2_token)

                if len(sp2_to_sp_token_indices) != 0:
                    scene_sp2_2_sp1_token_indices.append(sp2_to_sp_token_indices)

            tokens = torch.stack(tokens, dim=0)
            batch_tokens.append(tokens)
            scene_sp2_2_sp1_token_indices = torch.cat(scene_sp2_2_sp1_token_indices, dim=0)
            if record_mode == 'remain':
                self.batch_convert_remain_indices.clear()
                self.batch_convert_remain_indices.append(scene_sp2_2_sp1_token_indices)
            elif record_mode == 'mask':
                self.batch_convert_mask_indices.clear()
                self.batch_convert_mask_indices.append(scene_sp2_2_sp1_token_indices)

        return batch_tokens

    def remove_padding(self, padded_tokens: torch.Tensor, higher_full_super_indices: list, batch_token_indices: list):
        if self.batch_sp2_set is None:
            return padded_tokens
        else:
            if type(self.batch_sp2_set) == list:
                self.batch_sp2_set = torch.squeeze(self.batch_sp2_set[0], dim=0)

            assert padded_tokens.dim() == 4, "check padded_tokens' dim "
            scene_unpadded_tokens = []

            for i, sp2_index in enumerate(self.batch_sp2_set):
                sp2_to_sp_indices = torch.where(higher_full_super_indices[0] == sp2_index)[0]
                sp1_num = len(
                    torch.tensor([index for index in sp2_to_sp_indices if index in batch_token_indices[0]], dtype=int))
                local_tokens = padded_tokens[0, i, :sp1_num]
                scene_unpadded_tokens.append(local_tokens)

            return torch.vstack(scene_unpadded_tokens)

    def forward(self, full_features, sp_coords, full_super_indices_10, full_super_indices_21):
        batch_token_embed = self.token_embed(full_features, full_super_indices_10)
        batch_pos_embed = self.pos_embed(sp_coords)

        batch_mask_token_embed, batch_remain_token_embed, batch_mask_indices, batch_remain_indices = self.mask_token_embed(
            batch_token_embed)

        if isinstance(batch_remain_indices, list):
            batch_remain_indices = torch.unsqueeze(batch_remain_indices[0], dim=0)
        if isinstance(batch_mask_indices, list):
            batch_mask_indices = torch.unsqueeze(batch_mask_indices[0], dim=0)
        if isinstance(batch_pos_embed, list):
            batch_pos_embed = torch.unsqueeze(batch_pos_embed[0], dim=0)
        if isinstance(batch_remain_token_embed, list):
            batch_remain_token_embed = torch.unsqueeze(batch_remain_token_embed[0], dim=0)
        if isinstance(batch_mask_token_embed, list):
            batch_mask_token_embed = torch.unsqueeze(batch_mask_token_embed[0], dim=0)

        batch_remain_pos_embed = batch_pos_embed[:, batch_remain_indices[0]]
        batch_mask_pos_embed = batch_pos_embed[:, batch_mask_indices[0]]

        # batch_remain_token_embed = self.pad_embed(batch_token_embed, full_super_indices_21, batch_remain_indices)
        # batch_mask_token_embed = self.pad_embed(batch_token_embed, full_super_indices_21, batch_mask_indices)
        # batch_remain_pos_embed = self.pad_embed(batch_pos_embed,
        #                                         full_super_indices_21,
        #                                         batch_remain_indices,
        #                                         record_mode='remain')
        # batch_mask_pos_embed = self.pad_embed(batch_pos_embed,
        #                                       full_super_indices_21,
        #                                       batch_mask_indices,
        #                                       record_mode='mask')

        return (batch_remain_token_embed, batch_mask_token_embed), (batch_remain_pos_embed,
                                                                    batch_mask_pos_embed), (batch_remain_indices,
                                                                                            batch_mask_indices)


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = None,
                 output_dim: int = None,
                 act_layer=nn.GELU,
                 dropout_prob: float = 0.):

        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''
class Local_Self_Attention(nn.Module):

    def __init__(self,
                 input_dim: int,
                 head_num: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 qkv_bias: bool = False,
                 qk_scale: bool = None):

        super().__init__()
        self.head_num = head_num
        head_dim = input_dim // self.head_num
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # TODO init scale
        self.scale = qk_scale or head_dim**1  #-0.5
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape
        # qkv.shape = [B, N, P, 3, H, C/H] -> [3, B, N, H, P, C/H]
        qkv = self.qkv(x).reshape(B, N, P, 3, self.head_num, C // self.head_num).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(attn.shape, k.transpose(-2, -1).shape, v.shape)

        x = (attn @ v).transpose(2, 3).reshape(B, N, P, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''


class Attention(nn.Module):  # Global Self Attention

    def __init__(self,
                 input_dim: int,
                 head_num: int,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 qkv_bias: bool = False,
                 qk_scale: bool = None):

        super().__init__()
        self.head_num = head_num
        head_dim = input_dim // self.head_num
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # TODO init scale
        self.scale = qk_scale or head_dim**1  #-0.5
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, local_attn_mask=None):
        B, N, C = x.shape
        # qkv.shape = [B, N, 3, H, C/H] -> [3, B, H, N, C/H]
        qkv = self.qkv(x).reshape(B, N, 3, self.head_num, C // self.head_num).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if local_attn_mask != None:
            local_attn_mask *= -1000000.
            local_attn_mask = local_attn_mask.cuda()
            attn[:, :] += local_attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        input_dim: int,
        head_num: int,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        dropout_prob=0.,
        attn_drop=0.,
        droppath_prob=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):

        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)

        self.drop_path = DropPath(drop_prob=droppath_prob) if droppath_prob > 0. else nn.Identity()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, dropout_prob=dropout_prob)
        self.attn = Attention(input_dim=input_dim,
                              head_num=head_num,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=dropout_prob)
        # self.local_attn = Local_Self_Attention(input_dim=input_dim,
        #                                        head_num=head_num,
        #                                        qkv_bias=qkv_bias,
        #                                        qk_scale=qk_scale,
        #                                        attn_drop=attn_drop,
        #                                        proj_drop=dropout_prob)

    def forward(self, x, local_attn_mask=None):
        x = self.norm1(x)
        x = self.attn(x, local_attn_mask)
        x = x + self.drop_path(x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + self.drop_path(x)
        return x


class MAE_Encoder(nn.Module):

    def __init__(self,
                 token_embed_dim: int,
                 head_num: int,
                 depth: int = 5,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: bool = None,
                 dropout_prob: float = 0.,
                 attn_drop: float = 0.,
                 droppath_prob: float = 0.):

        super().__init__()
        droppath_prob = [p.item() for p in torch.linspace(0, droppath_prob, depth)]

        self.blocks = nn.ModuleList([
            Block(input_dim=token_embed_dim,
                  head_num=head_num,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  dropout_prob=dropout_prob,
                  attn_drop=attn_drop,
                  droppath_prob=droppath_prob[i] if isinstance(droppath_prob, list) else droppath_prob)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(token_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, local_attn_mask=None):
        for _, block in enumerate(self.blocks):
            x = block(x, local_attn_mask)
        x = self.norm(x)
        return x


class MAE_Decoder(nn.Module):

    def __init__(self,
                 token_embed_dim: int,
                 depth: int,
                 head_num: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: bool = None,
                 dropout_prob: float = 0.,
                 attn_drop: float = 0.,
                 droppath_prob: float = 0.1,
                 norm_layer=nn.LayerNorm):

        super().__init__()

        self.blocks = nn.ModuleList([
            Block(input_dim=token_embed_dim,
                  head_num=head_num,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  dropout_prob=dropout_prob,
                  attn_drop=attn_drop,
                  droppath_prob=droppath_prob[i] if isinstance(droppath_prob, list) else droppath_prob)
            for i in range(depth)
        ])
        self.norm = norm_layer(token_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, full_x, full_pos, pad_limit, sep):
        for _, block in enumerate(self.blocks):
            full_x = block(full_x + full_pos)

        rec_x = self.norm(full_x)
        rec_remain_x = rec_x[:, :sep]
        rec_mask_x = rec_x[:, sep:]
        # rec_remain_x = self.norm(full_x[:, :, :pad_limit])
        # rec_mask_x = self.norm(full_x[:, :, pad_limit:])
        return rec_remain_x, rec_mask_x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * np.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DAB_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DAB_TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 token_embed_dim,
                 head_num,
                 ffn_dim=512,
                 dropout_prob=0.1,
                 activation="relu",
                 normalize_before=False,
                 keep_query_pos=False):
        
        super().__init__()

        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.sa_qpos_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.sa_kcontent_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.sa_kpos_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.sa_v_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.self_attn = MultiheadAttention(token_embed_dim, head_num, dropout=dropout_prob, vdim=token_embed_dim)

        self.norm1 = nn.LayerNorm(token_embed_dim)
        self.dropout1 = nn.Dropout(dropout_prob)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.ca_qpos_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.ca_kcontent_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.ca_kpos_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.ca_v_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.ca_qpos_sine_proj = nn.Linear(token_embed_dim, token_embed_dim)
        self.cross_attn = MultiheadAttention(token_embed_dim*2, head_num, dropout=dropout_prob, vdim=token_embed_dim)

        self.head_num = head_num

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(token_embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(ffn_dim, token_embed_dim)

        
        self.norm2 = nn.LayerNorm(token_embed_dim)
        self.norm3 = nn.LayerNorm(token_embed_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,                           # coord pos embed
                query_pos: Optional[Tensor] = None,                     # query anchor embed
                query_sine_embed = None,
                is_first = False,
                seq_group_size=None):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries(1024) x batch_size(lv2 sp num) x embedding dimension(256)
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)
        
        # Sequential execution of attention
        tgt_group = torch.split(tgt, seq_group_size, 1)                 # tuples, each of shape: query_num x seq_group_size x embed_dim
        q_content_group = torch.split(q_content, seq_group_size, 1)
        q_pos_group = torch.split(q_pos, seq_group_size, 1)
        k_content_group = torch.split(k_content, seq_group_size, 1)
        k_pos_group = torch.split(k_pos, seq_group_size, 1)
        v_group = torch.split(v, seq_group_size, 1)

        group_num = int(np.ceil(tgt.shape[1]/seq_group_size))

        num_queries, bs, n_model = q_content.shape  # num_queries is the max_allowable_points/superpoint
        n, _, _ = k_content.shape

        processed_tgt = []

        for group_idx, group_data in enumerate(zip(tgt_group, q_content_group, q_pos_group, k_content_group, k_pos_group, v_group)):
            tgt_data = group_data[0]
            q_data = group_data[1]
            q_pos_data = group_data[2]
            k_data = group_data[3]
            k_pos_data = group_data[4]
            v_data = group_data[5]

            q = q_data + q_pos_data
            k = k_data + k_pos_data

            tgt2 = self.self_attn(q, k, value=v_data,
                                  attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]

            this_tgt = tgt_data + self.dropout1(tgt2)
            processed_tgt.append(this_tgt)

        # ========== End of Self-Attention =============

        tgt = torch.concat(processed_tgt, dim=1)
        tgt = self.norm1(tgt)
        print('here')

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        n, _, _ = k_content.shape

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)

        # Sequential execution of cross attention
        tgt_group = torch.split(tgt, seq_group_size, 1)
        q_content_group = torch.split(q_content, seq_group_size, 1)
        q_pos_group = torch.split(q_pos, seq_group_size, 1)
        k_content_group = torch.split(k_content, seq_group_size, 1)
        k_pos_group = torch.split(k_pos, seq_group_size, 1)
        v_group = torch.split(v, seq_group_size, 1)
        print(tgt_group[0].shape)
        exit()

        for group_idx, group_data in enumerate(zip(tgt_group, q_content_group, q_pos_group, k_content_group, k_pos_group, v_group)):
            tgt_data = group_data[0]
            q_data = group_data[1]
            q_pos_data = group_data[2]
            k_data = group_data[3]
            k_pos_data = group_data[4]
            v_data = group_data[5]

            if is_first or self.keep_query_pos:
                q = q_data + q_pos_data
                k = k_data + k_pos_data
            else:
                q = q_data
                k = k_data

            print(q.shape, k.shape)
            exit()

            q = q.view(num_queries, seq_group_size, self.head_num, n_model//self.head_num)
            query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed = query_sine_embed.view(num_queries, seq_group_size, self.head_num, n_model//self.v)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, seq_group_size, n_model * 2)
            k = k.view(n, seq_group_size, self.head_num, n_model//self.head_num)
            k_pos = k_pos.view(n, seq_group_size, self.head_num, n_model//self.head_num)
            k = torch.cat([k, k_pos], dim=3).view(n, seq_group_size, n_model * 2)

            tgt2 = self.cross_attn(query=q, key=k, value=v,
                                attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]

        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DAB_TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False, 
                 d_model=256,
                 query_dim=2,
                 keep_query_pos=False,
                 query_scale_type='cond_elewise',
                #  modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False):
        
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        # assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = DAB_MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = DAB_MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = DAB_MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        # self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        # if modulate_hw_attn:
        #     self.ref_anchor_head = DAB_MLP(d_model, d_model, 2, 2)
   
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                seq_group_size = None):
        
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2]

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)  
            query_pos = self.ref_point_head(query_sine_embed) 
    
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            # modulated HW attentions
            # if self.modulate_hw_attn:
            #     refHW_cond = self.ref_anchor_head(output).sigmoid()
            #     query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
            #     query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,                                       # None
                           memory_mask=memory_mask,                                 # None
                           tgt_key_padding_mask=tgt_key_padding_mask,               # None
                           memory_key_padding_mask=memory_key_padding_mask,         # TODO see MAE
                           pos=pos, 
                           query_pos=query_pos,
                           query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0),
                           seq_group_size=seq_group_size)
            print(output.shape)
            exit()

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)



@register_network('Superpoint_MAE_Pretrain')
class Superpoint_MAE(nn.Module):

    def __init__(self, config=None):

        super().__init__()
        # params
        self.mask_level = config['mask_level']
        self.mask_ratio = config['mask_ratio']
        self.feature_dim = config['feature_dim']
        self.pos_embed_hidden_dim = config['pos_embed_hidden_dim']
        self.token_embed_dim = config['token_embed_dim']
        self.pad_limit = config['token_pad_limit']
        self.head_num = config['head_num']
        self.mlp_ratio = config['MLP_ratio']
        self.dropout_prob = config['dropout_prob']
        self.droppath_prob = config['droppath_prob']
        self.encoder_depth = config['encoder_depth']
        self.decoder_depth = config['decoder_depth']

        # nn
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.token_embed_dim))
        self.prep_embed = Embed_and_Prep(sp_feature_dim=self.feature_dim,
                                         sp_embed_dim=self.token_embed_dim,
                                         pad_limit=self.pad_limit,
                                         mask_ratio=self.mask_ratio)
        self.encoder = MAE_Encoder(token_embed_dim=self.token_embed_dim,
                                   head_num=self.head_num,
                                   depth=self.encoder_depth,
                                   mlp_ratio=self.mlp_ratio,
                                   dropout_prob=self.dropout_prob,
                                   droppath_prob=self.droppath_prob)
        self.decoder = MAE_Decoder(token_embed_dim=self.token_embed_dim,
                                   head_num=self.head_num,
                                   depth=self.decoder_depth,
                                   mlp_ratio=self.mlp_ratio,
                                   dropout_prob=self.dropout_prob,
                                   droppath_prob=self.droppath_prob)
        
        # self.projector = nn.Linear(self.token_embed_dim, 11)
        self.projector = nn.Sequential(
            nn.Conv1d(self.token_embed_dim, 11*75, 1)
        )

        self.lossf1 = dist_chamfer_3D.chamfer_3DDist()
        self.lossf2 = nn.MSELoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, extras, **kwargs):
        full_features = extras['full_features']
        full_super_indices_21 = extras['full_super_indices_21']
        full_super_indices_20 = extras['full_super_indices_20']
        sp_coords = extras['sp2_coords']
        sp_attr = extras['sp2_attr']

        # currently operating at level 2

        assert len(full_features) and len(sp_coords) == 1, 'only batch size: 1 is currently supported'

        batch_token_embed, batch_pos_embed, indices = self.prep_embed(full_features, sp_coords, full_super_indices_20,
                                                                      full_super_indices_21)

        batch_remain_token_embed, batch_mask_token_embed = batch_token_embed[0], batch_token_embed[1]
        batch_remain_pos_embed, batch_mask_pos_embed = batch_pos_embed[0], batch_pos_embed[1]

        remain_x = self.encoder(batch_remain_token_embed)

        B, N_r, C = remain_x.shape
        _, N_m, _ = batch_mask_pos_embed.shape
        mask_x = self.mask_token.expand(B, N_m, -1)
        full_x = torch.cat((remain_x, mask_x), dim=1)
        full_pos = torch.cat((batch_remain_pos_embed, batch_mask_pos_embed), dim=1)

        rec_remain_x, rec_mask_x = self.decoder(full_x, full_pos, self.pad_limit, sep=N_r)

        rec_full_x = torch.cat((rec_remain_x, rec_mask_x), dim=1)

        # rec_x = self.projector(rec_full_x)
        rec_x = self.projector(rec_full_x.transpose(1,2)).transpose(1,2).reshape(B, N_r+N_m, 75, 11)

        rec_x_indices = torch.cat((indices[0], indices[1]), dim=1)
        sort = torch.squeeze(torch.argsort(rec_x_indices, dim=1))
        # assert len(rec_x) == len(rec_x_indices), f'{rec_x.shape, rec_x_indices.shape}'
        rec_x = torch.squeeze(rec_x)[sort]

        rec_x = rec_x.reshape((N_r+N_m)*75, 11)

        rec_x_indices = torch.squeeze(rec_x_indices)[sort]
        rec_x_coords = rec_x[:, :3]
        rec_x_attr = rec_x[:, 3:]

        original_sort = torch.argsort(full_super_indices_20[0])
        original_data = full_features[0][original_sort]
        original_coords = original_data[:, :3]
        original_attr = original_data[:, 3:]

        # Loss
        target1, target2 = original_coords, original_attr
        dist1, dist2, _, _ = self.lossf1(target1.unsqueeze(0).type(torch.float32), rec_x_coords.unsqueeze(0).type(torch.float32))
        loss1 = torch.mean(dist1**2) + torch.mean(dist2**2)
        # loss2 = self.lossf2(rec_x_attr, target2)
        loss2 = 0.
        loss = loss1 + loss2

        '''
        B, N_2, P, C = remain_x.shape
        mask_x = torch.zeros((B, N_2, P, C)).cuda()
        full_x = torch.cat((remain_x, mask_x), dim=2)
        full_pos = torch.cat((batch_remain_pos_embed, batch_mask_pos_embed), dim=2)

        rec_remain_x, rec_mask_x = self.decoder(full_x, full_pos, self.pad_limit)

        rec_remain_x = self.prep_embed.remove_padding(rec_remain_x, full_super_indices_21, indices[0])
        rec_mask_x = self.prep_embed.remove_padding(rec_mask_x, full_super_indices_21, indices[1])

        rec_full_x = torch.cat((rec_remain_x, rec_mask_x), dim=0)

        rec_x_coords = self.projector(rec_full_x)

        rec_x_indices = torch.cat(
            (self.prep_embed.batch_convert_remain_indices[0], self.prep_embed.batch_convert_mask_indices[0]), dim=0)
        sort = torch.argsort(rec_x_indices)

        assert len(rec_x_coords) == len(rec_x_indices), f'{rec_x_coords.shape, rec_x_indices.shape}'

        rec_x_coords = rec_x_coords[sort]
        rec_x_indices = rec_x_indices[sort]
        '''

        # Loss
        # target1, target2 = sp_coords[0], sp_attr[0].type(torch.float32)
        # dist1, dist2, _, _ = self.lossf1(target1.unsqueeze(0).cuda(), rec_x_coords.unsqueeze(0))
        # loss1 = torch.mean(dist1**2) + torch.mean(dist2**2)
        # loss2 = self.lossf2(rec_x_attr, target2)
        # loss = loss1 + loss2

        return loss


@register_network('Superpoint_MAE_DAB_Pretrain')
class Superpoint_MAE_DAB(nn.Module):

    def __init__(self, config=None):

        super().__init__()
        # params
        self.mask_level = config['mask_level']
        self.mask_ratio = config['mask_ratio']
        self.feature_dim = config['feature_dim']
        self.pos_embed_hidden_dim = config['pos_embed_hidden_dim']
        self.token_embed_dim = config['token_embed_dim']
        self.pad_limit = config['token_pad_limit']
        self.head_num = config['head_num']
        self.mlp_ratio = config['MLP_ratio']
        self.dropout_prob = config['dropout_prob']
        self.droppath_prob = config['droppath_prob']
        self.encoder_depth = config['encoder_depth']
        self.decoder_depth = config['decoder_depth']

        self.ffn_dim = config['ffn_dim']
        self.query_num = config['query_num']
        self.query_dim = config['query_dim']        # x,y
        self.group_size = config['group_size']

        # nn
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.token_embed_dim))
        self.prep_embed = Embed_and_Prep(sp_feature_dim=self.feature_dim,
                                         sp_embed_dim=self.token_embed_dim,
                                         pad_limit=self.pad_limit,
                                         mask_ratio=self.mask_ratio)
        self.encoder = MAE_Encoder(token_embed_dim=self.token_embed_dim,
                                   head_num=self.head_num,
                                   depth=self.encoder_depth,
                                   mlp_ratio=self.mlp_ratio,
                                   dropout_prob=self.dropout_prob,
                                   droppath_prob=self.droppath_prob)
        
        # self.decoder = MAE_Decoder(token_embed_dim=self.token_embed_dim,
        #                            head_num=self.head_num,
        #                            depth=self.decoder_depth,
        #                            mlp_ratio=self.mlp_ratio,
        #                            dropout_prob=self.dropout_prob,
        #                            droppath_prob=self.droppath_prob)

        
        self.refpoint_embed = nn.Embedding(self.query_num, self.query_dim)

        DAB_decoder_layer = DAB_TransformerDecoderLayer(token_embed_dim=self.token_embed_dim,
                                                        head_num=self.head_num,
                                                        ffn_dim=self.ffn_dim)

        decoder_norm = nn.LayerNorm(self.token_embed_dim)

        self.DAB_decoder = DAB_TransformerDecoder(decoder_layer=DAB_decoder_layer,
                                                  num_layers=self.decoder_depth,
                                                  norm=decoder_norm,
                                                  return_intermediate=False,
                                                  d_model=self.token_embed_dim,
                                                  query_dim=self.query_dim,
                                                  keep_query_pos=False,
                                                #   modulate_hw_attn=True,
                                                  bbox_embed_diff_each_layer=False)

        
        # self.projector = nn.Linear(self.token_embed_dim, 11)
        self.projector = nn.Sequential(
            nn.Conv1d(self.token_embed_dim, 11*75, 1)
        )

        self.lossf1 = dist_chamfer_3D.chamfer_3DDist()
        self.lossf2 = nn.MSELoss()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, extras, **kwargs):
        full_features = extras['full_features']
        full_super_indices_21 = extras['full_super_indices_21']
        full_super_indices_20 = extras['full_super_indices_20']
        sp_coords = extras['sp2_coords']
        sp_attr = extras['sp2_attr']

        # currently operating at level 2
        assert len(full_features) and len(sp_coords) == 1, 'only batch size: 1 is currently supported'

        # =====Embedding=====
        batch_token_embed, batch_pos_embed, indices = self.prep_embed(full_features, sp_coords, full_super_indices_20,
                                                                      full_super_indices_21)
        batch_remain_token_embed, batch_mask_token_embed = batch_token_embed[0], batch_token_embed[1]
        batch_remain_pos_embed, batch_mask_pos_embed = batch_pos_embed[0], batch_pos_embed[1]

        # =====Transformer Encoder=====
        remain_x = self.encoder(batch_remain_token_embed)

        B, N_r, C = remain_x.shape
        _, N_m, _ = batch_mask_pos_embed.shape
        mask_x = self.mask_token.expand(B, N_m, -1)
        full_x = torch.cat((remain_x, mask_x), dim=1)
        full_pos = torch.cat((batch_remain_pos_embed, batch_mask_pos_embed), dim=1)

        # =====Sort Encoder Output=====
        rec_indices = torch.cat((indices[0], indices[1]), dim=1)
        sort = torch.squeeze(torch.argsort(rec_indices, dim=1))
        full_x = torch.squeeze(full_x)[sort]
        full_pos = torch.squeeze(full_pos)[sort]

        print(f'memory: {full_x.shape}')

        # =====Transformer Decoder (DAB)=====
        tgt = torch.zeros(self.query_num, N_r+N_m, self.token_embed_dim).cuda()        # batch size => superpoint num at lv2
        print(f'target: {tgt.shape}')

        embedweight = self.refpoint_embed.weight
        refpoint_embed = embedweight.unsqueeze(1).repeat(1, N_r+N_m, 1)
        print(f'refpoint_embed: {refpoint_embed.shape}')

        out = self.DAB_decoder(tgt=tgt,
                               memory=full_x,
                               memory_key_padding_mask=None,                #TODO what is the purpose of masking in cross attention?
                               pos=full_pos,
                               refpoints_unsigmoid=refpoint_embed,
                               seq_group_size=self.group_size)
        exit()

        '''
        rec_remain_x, rec_mask_x = self.decoder(full_x, full_pos, self.pad_limit, sep=N_r)

        rec_full_x = torch.cat((rec_remain_x, rec_mask_x), dim=1)

        # rec_x = self.projector(rec_full_x)
        rec_x = self.projector(rec_full_x.transpose(1,2)).transpose(1,2).reshape(B, N_r+N_m, 75, 11)

        rec_x_indices = torch.cat((indices[0], indices[1]), dim=1)
        sort = torch.squeeze(torch.argsort(rec_x_indices, dim=1))
        # assert len(rec_x) == len(rec_x_indices), f'{rec_x.shape, rec_x_indices.shape}'
        rec_x = torch.squeeze(rec_x)[sort]

        rec_x = rec_x.reshape((N_r+N_m)*75, 11)

        rec_x_indices = torch.squeeze(rec_x_indices)[sort]
        rec_x_coords = rec_x[:, :3]
        rec_x_attr = rec_x[:, 3:]

        original_sort = torch.argsort(full_super_indices_20[0])
        original_data = full_features[0][original_sort]
        original_coords = original_data[:, :3]
        original_attr = original_data[:, 3:]

        # Loss
        target1, target2 = original_coords, original_attr
        dist1, dist2, _, _ = self.lossf1(target1.unsqueeze(0).type(torch.float32), rec_x_coords.unsqueeze(0).type(torch.float32))
        loss1 = torch.mean(dist1**2) + torch.mean(dist2**2)
        # loss2 = self.lossf2(rec_x_attr, target2)
        loss2 = 0.
        loss = loss1 + loss2
        '''

        return loss
    