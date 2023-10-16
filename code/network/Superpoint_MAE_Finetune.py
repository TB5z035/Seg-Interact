import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import register_network


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

    def __init__(self, sp_feature_dim: int, sp_embed_dim: int, pad_limit: int = 64):

        super().__init__()
        self.pad_limit = pad_limit
        self.batch_sp2_set = None
        self.batch_convert_remain_indices = []
        self.batch_convert_mask_indices = []
        self.batch_convert_full_indices = []

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

    def pad_embed(self, batch_token_embed, higher_full_super_indices, batch_indices=None, record_mode=''):
        assert record_mode in ['', 'remain', 'mask', 'full'], 'check pad_embed record_mode'
        batch_tokens = []
        self.batch_sp2_set = []

        for i in range(len(higher_full_super_indices)):
            sp2_set = torch.unique(higher_full_super_indices[i])
            self.batch_sp2_set.append(sp2_set)
            tokens = []
            scene_sp2_2_sp1_token_indices = []

            for sp2_index in sp2_set:
                sp2_to_sp_indices = torch.where(higher_full_super_indices[i] == sp2_index)[0]
                if batch_indices != None:
                    for j in range(len(batch_indices)):
                        batch_indices[j] = batch_indices[j].cuda()
                    sp2_to_sp_token_indices = torch.tensor(sorted(
                        [index for index in sp2_to_sp_indices if index in batch_indices[i]]),
                                                           dtype=int)

                else:
                    sp2_to_sp_token_indices = sp2_to_sp_indices
                sp2_token = batch_token_embed[i][sp2_to_sp_token_indices]
                assert sp2_token.dim() == 2, f'sp2_remain_token has dim: {sp2_token.dim()} which is invalid for padding'
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
            elif record_mode == 'full':
                self.batch_convert_full_indices.clear()
                self.batch_convert_full_indices.append(scene_sp2_2_sp1_token_indices)

        return batch_tokens

    def remove_padding(self,
                       padded_tokens: torch.Tensor,
                       higher_full_super_indices: list,
                       batch_token_indices: list = None):
        if self.batch_sp2_set is None:
            return padded_tokens
        else:
            if type(self.batch_sp2_set) == list:
                self.batch_sp2_set = torch.squeeze(self.batch_sp2_set[0], dim=0)

            assert padded_tokens.dim() == 4, "check padded_tokens' dim "
            scene_unpadded_tokens = []

            for i, sp2_index in enumerate(self.batch_sp2_set):
                sp2_to_sp_indices = torch.where(higher_full_super_indices[0] == sp2_index)[0]
                if batch_token_indices != None:
                    sp1_num = len(
                        torch.tensor([index for index in sp2_to_sp_indices if index in batch_token_indices[0]],
                                     dtype=int))
                else:
                    sp1_num = len(sp2_to_sp_indices)
                local_tokens = padded_tokens[0, i, :sp1_num]
                scene_unpadded_tokens.append(local_tokens)

            return torch.vstack(scene_unpadded_tokens)

    def forward(self, full_features, sp_coords, full_super_indices_10, full_super_indices_21):
        batch_token_embed = self.token_embed(full_features, full_super_indices_10)
        batch_pos_embed = self.pos_embed(sp_coords)

        batch_token_embed = self.pad_embed(batch_token_embed, full_super_indices_21)
        batch_pos_embed = self.pad_embed(batch_pos_embed, full_super_indices_21, record_mode='full')

        if isinstance(batch_token_embed, list):
            batch_token_embed = torch.unsqueeze(batch_token_embed[0], dim=0)
        if isinstance(batch_pos_embed, list):
            batch_pos_embed = torch.unsqueeze(batch_pos_embed[0], dim=0)

        return batch_token_embed, batch_pos_embed


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


class Attention(nn.Module):

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
        # qkv.shape = [3, B, N, H, P, C/H]
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


class Block(nn.Module):

    def __init__(self,
                 input_dim: int,
                 head_num: int,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout_prob=0.,
                 attn_drop=0.,
                 droppath_prob=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)

        self.drop_path = DropPath(drop_prob=droppath_prob) if droppath_prob > 0. else nn.Identity()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, dropout_prob=dropout_prob)
        self.local_attn = Attention(input_dim=input_dim,
                                    head_num=head_num,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=dropout_prob)

    def forward(self, x):
        x = self.norm1(x)
        x = self.local_attn(x)
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

    def forward(self, x, pos_embed):
        for _, block in enumerate(self.blocks):
            x = block(x + pos_embed)
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

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        x = self.norm(x)
        return x


class MAE_Unsample(nn.Module):

    def __init__(self):

        super().__init__()

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """

        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src.type(torch.float), dst.permute(0, 2, 1).type(torch.float))
        dist += torch.sum(src**2, -1).view(B, N, 1)
        dist += torch.sum(dst**2, -1).view(B, 1, M)
        return dist

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def forward(self, higher_feats, lower_points, higher_points):
        """
        Input:
            higher_feats: [B, N_sp2,C]
            lower_points: [B, N_points, 3]
            higher_points: [B, N_sp1, 3]
        Return:
            interpolated_feats: [B, N_points, C]
        """

        B, N, _ = lower_points.shape
        _, M, _ = higher_points.shape

        dists = self.square_distance(lower_points, higher_points)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = torch.sum(self.index_points(higher_feats, idx) * weight.view(B, N, 3, 1), dim=2)
        return interpolated_feats


@register_network('Superpoint_MAE_Finetune')
class Superpoint_MAE(nn.Module):

    def __init__(self, config=None):

        super().__init__()
        # params
        self.mask_level = config['mask_level']
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
        self.cls_num = config['class_num']

        # nn
        self.prep_embed = Embed_and_Prep(sp_feature_dim=self.feature_dim,
                                         sp_embed_dim=self.token_embed_dim,
                                         pad_limit=self.pad_limit)
        self.encoder = MAE_Encoder(token_embed_dim=self.token_embed_dim,
                                   head_num=self.head_num,
                                   depth=self.encoder_depth,
                                   mlp_ratio=self.mlp_ratio,
                                   dropout_prob=self.dropout_prob,
                                   droppath_prob=self.droppath_prob)
        self.unsampling = MAE_Unsample()
        self.decoder = MAE_Decoder(token_embed_dim=self.token_embed_dim,
                                   head_num=self.head_num,
                                   depth=self.decoder_depth,
                                   mlp_ratio=self.mlp_ratio,
                                   dropout_prob=self.dropout_prob,
                                   droppath_prob=self.droppath_prob)
        self.cls_projector = nn.Linear(self.token_embed_dim, self.cls_num)
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, extras, **kwargs):
        full_features = extras['full_features']
        full_super_indices_10 = extras['full_super_indices_10']
        full_super_indices_21 = extras['full_super_indices_21']
        print('sp1_num:', len(full_super_indices_21[0]))
        sp1_coords = extras['sp1_coords']
        assert len(full_features) and len(sp1_coords) == 1, 'only batch size: 1 is currently supported'
        batch_token_embed, batch_pos_embed = self.prep_embed(full_features, sp1_coords, full_super_indices_10,
                                                             full_super_indices_21)
        x = self.encoder(batch_token_embed, batch_pos_embed)
        x = self.decoder(x, batch_pos_embed)

        x = self.prep_embed.remove_padding(x, full_super_indices_21)
        x_indices = self.prep_embed.batch_convert_full_indices[0]
        sort = torch.argsort(x_indices)
        assert len(x) == len(x_indices), f'{x.shape, x_indices.shape}'
        x, x_indices = x[sort], x_indices[sort]
        original_coords = full_features[0][:, :3]

        unsampled_x = self.unsampling(torch.unsqueeze(x, dim=0), torch.unsqueeze(original_coords, dim=0),
                                      torch.unsqueeze(sp1_coords[0], dim=0))
        logits = self.cls_projector(unsampled_x)
        scores = self.softmax(logits)

        return scores
