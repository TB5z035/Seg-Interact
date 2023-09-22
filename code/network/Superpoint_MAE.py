import numpy as np
import torch

from . import register_network

import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm
# from timm.models.layers import DropPath, trunc_normal_
import numpy as np

# from .build import MODELS
# from utils import misc
import random
# from knn_cuda import KNN
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class Encoder(nn.Module):  ## Embedding module

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                        nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                         nn.Conv1d(512, self.encoder_channel, 1))

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


## Transformers


class TransformerEncoder(nn.Module):

    def __init__(self,
                 embed_dim=768,
                 depth=4,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self,
                 embed_dim=384,
                 depth=4,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


# original code
class Point_MAE(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1))

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis:  #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1


# finetune model
# @MODELS.register_module()
class PointTransformer(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(nn.Linear(self.trans_dim * 2, 256), nn.BatchNorm1d(256),
                                               nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 256),
                                               nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                               nn.Linear(256, self.cls_dim))

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger='Transformer')
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='Transformer')

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


class SuperPointNet(nn.Module):

    def __init__(self,
                 sp_feature_dim,
                 sp_embed_dim,
                 hidden_dim=64,
                 pad_limit=64,
                 mask_ratio=0.6):

        super().__init__()
        self.pad_limit = pad_limit
        self.mask_ratio = mask_ratio

        self.sp_linear_1 = nn.Linear(sp_feature_dim, hidden_dim)
        self.sp_linear_2 = nn.Linear(hidden_dim, sp_embed_dim)

    def gen_token_embed(self, full_features, full_super_indices):
        sp_features_batch = []
        for i, full_feature in enumerate(full_features):
            # print(f'batch: {i}')
            sp_max = torch.max(full_super_indices[i])
            sp_features = []
            for sp_index in range(sp_max + 1):
                point_features = full_feature[torch.where(full_super_indices[i] == sp_index)[0]].to(torch.float32)
                point_features = F.relu(self.sp_linear_1(point_features.cuda()))
                point_features = F.relu(self.sp_linear_2(point_features))
                sp_feature = torch.max(point_features, dim=0, keepdim=True)[0]
                sp_features.append(sp_feature)
            sp_features = torch.cat(sp_features, dim=0)
            sp_features_batch.append(sp_features)
        return sp_features_batch

    def mask_token_embed(self, batch_token_embed):
        batch_mask_embed = []
        batch_remain_embed = []
        batch_mask_indices = []
        batch_remain_indices = []

        for batch_index in range(len(batch_token_embed)):
            N, C = batch_token_embed[batch_index].shape
            mask_num = int(N * self.mask_ratio)
            mask_indices = torch.from_numpy(np.random.choice(N, mask_num, replace=False))
            remain_indices = torch.tensor([num for num in range(N) if num not in mask_indices])
            batch_mask_embed.append(batch_token_embed[batch_index][mask_indices])
            batch_remain_embed.append(batch_token_embed[batch_index][remain_indices])
            batch_mask_indices.append(mask_indices)
            batch_remain_indices.append(remain_indices)

        return batch_mask_embed, batch_remain_embed, batch_mask_indices, batch_remain_indices

    def pad_token_embed(self, batch_token_embed, higher_full_super_indices, batch_remain_indices, batch_mask_indices):
        batch_remain_tokens = []
        batch_mask_tokens = []
        for i in range(len(higher_full_super_indices)):
            sp2_set = set(higher_full_super_indices[i])

            remain_tokens = []
            mask_tokens = []
            for sp2_index in sp2_set:
                sp2_to_sp_indices = torch.where(higher_full_super_indices[i] == sp2_index)[0]
                sp2_to_sp_remain_indices = torch.tensor(
                    [index for index in sp2_to_sp_indices if index not in batch_mask_indices[i]], dtype=int)
                sp2_to_sp_mask_indices = torch.tensor(
                    [index for index in sp2_to_sp_indices if index in batch_mask_indices[i]], dtype=int)
                sp2_remain_token = batch_token_embed[i][sp2_to_sp_remain_indices]
                sp2_mask_token = batch_token_embed[i][sp2_to_sp_mask_indices]
                assert sp2_remain_token.dim(
                ) == 2, f'sp2_remain_token has dim: {sp2_remain_token.dim()} which is invalid for padding'
                assert sp2_mask_token.dim(
                ) == 2, f'sp2_mask_token has dim: {sp2_mask_token.dim()} which is invalid for padding'
                remain_pad_dim, mask_pad_dim = self.pad_limit - sp2_remain_token.shape[
                    0], self.pad_limit - sp2_mask_token.shape[0]
                sp2_remain_token, sp2_mask_token = F.pad(sp2_remain_token, (0, 0, 0, remain_pad_dim)), F.pad(
                    sp2_mask_token, (0, 0, 0, mask_pad_dim))
                remain_tokens.append(sp2_remain_token)
                mask_tokens.append(sp2_mask_token)
            remain_tokens, mask_tokens = torch.stack(remain_tokens, dim=0), torch.stack(mask_tokens, dim=0)
            batch_remain_tokens.append(remain_tokens)
            batch_mask_tokens.append(mask_tokens)
        return batch_remain_tokens, batch_mask_tokens

    def forward(self, full_features, full_super_indices_10, full_super_indices_21):
        batch_token_embed = self.gen_token_embed(full_features, full_super_indices_10)
        _, _, batch_mask_indices, batch_remain_indices = self.mask_token_embed(batch_token_embed)
        batch_remain_token_embed, batch_mask_token_embed = self.pad_token_embed(batch_token_embed,
                                                                                full_super_indices_21,
                                                                                batch_remain_indices,
                                                                                batch_mask_indices)
        print(batch_remain_token_embed[0].shape, batch_mask_token_embed[0].shape)
        return batch_remain_token_embed, batch_mask_token_embed


class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 act_layer=nn.GELU,
                 drop=0.):
        
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 num_heads=4,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 qk_scale=None):

        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # TODO init scale
        self.scale = qk_scale or head_dim**1  #-0.5
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        assert len(x) == 1, 'only support batch size = 1'
        x = torch.unsqueeze(x[0], dim=0) if type(x) != torch.Tensor else x
        B, N, P, C = x.shape
        # qkv.shape = [3, B, N, H, P, C/H]
        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                                    C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
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
                 input_dim,
                 head_num,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(in_dim=input_dim,
                       hidden_dim=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.attn = Attention(input_dim=input_dim,
                              num_heads=head_num,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@register_network('Superpoint_MAE')
class Superpoint_MAE(nn.Module):

    def __init__(self, config=None):

        super().__init__()
        # params
        self.mask_level = config['mask_level']
        self.mask_ratio = config['mask_ratio']
        self.feature_dim = config['feature_dim']
        self.token_embed_dim = config['token_embed_dim']
        self.pad_limit = config['token_pad_limit']
        self.head_num = config['head_num']
        self.mlp_ratio = config['MLP_ratio']

        # nn
        self.point_net = SuperPointNet(sp_feature_dim=self.feature_dim,
                                       sp_embed_dim=self.token_embed_dim,
                                       pad_limit=self.pad_limit,
                                       mask_ratio=self.mask_ratio)
        
        self.block = Block(input_dim=self.token_embed_dim,
                           head_num=self.head_num,
                           mlp_ratio=self.mlp_ratio)

        self.local_attention = Attention(input_dim=self.token_embed_dim)
        self.mlp = MLP(input_dim=self.token_embed_dim,
                       hidden_dim=self.token_embed_dim,
                       output_dim=self.token_embed_dim)

    def forward(self, inputs, extras, **kwargs):
        full_features = extras['full_features']
        full_super_indices_10 = extras['full_super_indices_10']
        full_super_indices_21 = extras['full_super_indices_21']

        batch_remain_token_embed, batch_mask_token_embed = self.point_net(full_features, full_super_indices_10,
                                                                          full_super_indices_21)

        x = self.local_attention(batch_remain_token_embed)
        x = self.mlp(x)
        print(x.shape)

        exit()
        return None


'''Stage 2


class MPN_MLP(nn.module):
    def __init__(self,
                 mlp_in_dim=8):
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
    def __init__(self,
                 encoder_in_dim=10,
                 encoder_out_dim=8):
        super().__init__()
        self.encoder_in_dim = encoder_in_dim
        self.encoder_out_dim = encoder_out_dim
        self.qkv = nn.Linear(self.encoder_in_dim, self.encoder_out_dim)

    def forward(self, point_features):
        return encoder_features


@register_network('MPN')
class MPN(MPN_Encoder, MPN_MLP):
    def __init__(self,
                 dims):
        self.encoder = MPN_Encoder()
        self.mlp = MPN_MLP()

    def forward(self):
        return mask_sp_indices

'''
