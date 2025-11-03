# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from .common_utils_mamba import *
import torch.nn.init as init

##=================================
class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Linear(d_model, d_expand, bias=bias)
        self.p_proj = nn.Linear(d_model, d_expand , bias=bias)
        self.act: nn.Module = act_layer()##nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)##nn.Linear(d_expand, d_model, bias=bias)##
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        # if not channel_first:
        #     x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        B,H,W,C = x.shape
        x = self.act(x).permute(0, 3, 1, 2).contiguous()
        ys = self.forward_core(x)
        return ys

##==============================
class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        look_forward_twice=False,
        mixed_selection=False,
        use_checkpoint=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers, use_checkpoint
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            look_forward_twice,
            use_checkpoint,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self.mixed_selection = mixed_selection
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N_, 1, 1, 2
            )
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, srcs, masks, pos_embeds, query_embed=None, self_attn_mask=None):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory
            )
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
                + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            self_attn_mask,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return hs, init_reference_out, inter_references_out, None, None

def ceil_avg_pool2d(input_tensor, kernel_size, stride):
    # 获取输入张量的高度和宽度
    batch_size, channels, height, width = input_tensor.shape

    # 计算需要填充的高度和宽度
    pad_height = (stride - (height % stride)) % stride
    pad_width = (stride - (width % stride)) % stride

    # 创建填充层
    padding_layer = nn.ZeroPad2d((0, pad_width, 0, pad_height))

    # 对输入张量进行填充
    padded_input = padding_layer(input_tensor)

    # 创建平均池化层
    avg_pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    # 进行池化操作
    output = avg_pool_layer(padded_input)

    return output
def bilinear_upsampling(input_tensor, target_size):
    """
    使用双线性插值进行上采样
    :param input_tensor: 输入张量
    :param scale_factor: 缩放因子
    :return: 上采样后的张量
    """
    upsampler = F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=True)
    return upsampler
##多尺度Mamba
class Ms_Mamba(nn.Module):
    def __init__(self,d_model,n_levels):
        super().__init__()
        self.norm_Mamba = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.ss2ds = nn.ModuleList()
        for _ in range(n_levels):
            ss2d = SS2D(d_model=d_model,
                        d_state=16,##===========
                        ssm_ratio=1.0,
                        ssm_rank_ratio=1.0,
                        dt_rank="auto",
                        act_layer=nn.SiLU,
                        d_conv=3,
                        conv_bias=True,
                        dropout= 0.0,
                        initialize="v0",
                        forward_type="v2",) ##=========
            self.ss2ds.append(ss2d)
        self.ant = nn.ModuleList()
        for _ in range(12):
            convs = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3,
                                stride=1, padding=1,groups=d_model),
                nn.Conv2d(d_model, d_model, kernel_size=1,
                                stride=1, padding=0,groups=1)
            ) 
            self.ant.append(convs)
        self.ans = nn.ModuleList()
        for _ in range(4):
            convs = nn.Conv2d(d_model, d_model, kernel_size=3,
                                stride=2, padding=1)
            self.ans.append(convs)
        self.anl = nn.ModuleList()
        for _ in range(4):
            self.anl.append(nn.Conv2d(d_model*2, d_model, kernel_size=1,stride=1, padding=0))##nn.Linear(d_model*2 , d_model)
        
        self.sig = nn.Sigmoid()
        self.out_norm = nn.LayerNorm(d_model)
        self._init_ss2d()
    def _init_ss2d(self):
        for ss2d in self.ss2ds:
            for name, param in ss2d.named_parameters():
                if "weight" in name:
                    if param.dim() >= 2:
                        # 对二维及以上权重使用 Xavier 初始化
                        init.xavier_uniform_(param)
                    else:
                        # 对一维权重使用正态分布初始化（例如状态参数）
                        init.normal_(param, mean=0.0, std=0.02)
                elif "bias" in name:
                    # 偏置初始化为零
                    init.zeros_(param)
 
    def forward(self,src2,spatial_shapes):
        hw_shape = spatial_shapes[:,0]*spatial_shapes[:,1]
        MSD_list = torch.split(src2, hw_shape.tolist() , dim=1)
        x_ss2d = []
        x_global = []
        for i,(x_MSD) in enumerate(MSD_list):
            B,N,C = x_MSD.shape
            H,W = spatial_shapes[i,:]
            x_MSD = x_MSD.view(B,H,W,C).contiguous()
            ys = self.ss2ds[i](x_MSD)
            x_ss2d.append(ys)   
        x_globals = (torch.cat(x_ss2d,1))

        return x_globals
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        ##SS2D
        self.ss2ds = Ms_Mamba(d_model,n_levels)
        self.norm_Mamba = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # 初始化参数
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
        t=6,
    ):

 ##==================================================1.0
        if t>=0 :   
            src2 = self.self_attn(
                self.with_pos_embed(src, pos),
                reference_points,
                src,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
            x_globals = self.ss2ds(src2,spatial_shapes)  
            src2 = self.norm_Mamba(x_globals+src2)
        else:
            src2 = self.self_attn(
                self.with_pos_embed(src, pos),
                reference_points,
                src,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                )
            else:
                output = layer(
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                    _,
                )
        # assert 1 is 0
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    self_attn_mask,
                )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries_one2one + args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.use_checkpoint,
    )

