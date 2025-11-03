# ------------------------------------------------------------------------
# DDM-DETR
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import SwinTransformer
from timm.models.layers import trunc_normal_
# from .ScConv import ScConv as scconv


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class TransformerBackbone(nn.Module):
    def __init__(
        self, backbone: str, train_backbone: bool, return_interm_layers: bool, args
    ):
        super().__init__()
        out_indices = (1, 2, 3)##(0, 1, 2, 3)
        if backbone == "swin_tiny":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_small":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_large":
            backbone = SwinTransformer(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 192
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_large_window12":
            backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 192
            backbone.init_weights(args.pretrained_backbone_path)
        else:
            raise NotImplementedError

        for name, parameter in backbone.named_parameters():
            # TODO: freeze some layers?
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:

            self.strides = [8, 16, 32]
            self.num_channels = [
                embed_dim * 2,
                embed_dim * 4,
                embed_dim * 8,
            ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]

        self.body = backbone
        
         ##cnn卷积操作：
        dims=[96,192,384,768]
        pre_dims=[96,96,192,384]
        self.norm_layer_res = nn.LayerNorm
        self.stages = nn.ModuleList()
        num_layer = [2,2,6,2]##2,3,4,3
        for i in range(4):
            layer = GCNet(
                Bottleneck,##卷积层的构建
                dims[i],##通道数
                num_layer[i],##卷积层的数量
                pre_dims[i],##降采样
                norm_layer=self.norm_layer_res,
            )
            self.stages.append(layer)

        ##===================================空间选择机制
        self.ssf_stages = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(3):
            ssf = SSF(dims[i+1])
            self.ssf_stages.append(ssf)
            # self.norm.append(nn.LayerNorm(dims[i]))

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        xs_n = {}
        out = 0
        for i, (name, x) in enumerate(xs.items()):
            if i<3:
                xs_n[name] = x
            else:
                out = x
        #  ##卷积操作
        xs_t = {}
        for i in range(4):
            if i == 0:
                x_cnn = self.stages[i](out)
            else:
                x_cnn = self.stages[i](x_cnn)##cnn
                # x_c = self.ssf_stages[i](x_cnn,x)
                xs_t[str(i)] = x_cnn
         ##融合操作
        xs_p = {}
        for i, (name, x) in enumerate(xs_n.items()):
            x_cnn = xs_t[name]
            x = xs_n[name]
            x_c = self.ssf_stages[i](x_cnn,x)
            xs_p[name] = x_c
        
        out: Dict[str, NestedTensor] = {}
        for name, x in xs_p.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if "resnet" in args.backbone:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation,
        )
    else:
        backbone = TransformerBackbone(
            args.backbone, train_backbone, return_interm_layers, args
        )
    model = Joiner(backbone, position_embedding)
    return model

##cnn卷积操作
class GCNet(nn.Module):
    def __init__(
        self,
        block,
        planes: int,  ##通道数
        blocks: int,  ##block的数量
        dim,
        norm_layer,
        dilation=1,
) :
        super().__init__()
        previous_dilation = dilation
        self.dim = dim
        self.plances = planes
        self.downsample_layer = PatchMerging(dim=dim, norm_layer=nn.LayerNorm)

        self.groups = 1
        self.layers = nn.ModuleList()
        for i in range(0, blocks):
            self.layers.append(
                block(
                    planes,
                    planes,
                    self.groups,
                    previous_dilation,
                    norm_layer,
                )
            )
        self._initialize_weights()
    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        if self.dim != self.plances:
            B,C,H,W = x.shape
            x = x.flatten(2).permute(0,2,1)##B,N,C
            x = self.downsample_layer(x,H,W)
            if H % 2 == 1:
                H += 1
            if W % 2 == 1:
                W += 1
            x = x.view(B, H // 2, W // 2, C*2).permute(0, 3, 1, 2).contiguous()  ##B,C,H,W
        for layer in self.layers:
            x = layer(x)
        return x

##下采样
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        ##输入图像的分辨率,也即是输入图像的宽和高
        # self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=True)
        self.norm = norm_layer(2 * dim)

    def forward(self, x,H,W):
        # H, W = self.input_resolution
        B, L,C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :] ##偶数行，偶数列  B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)## B H/2 W/2 4C
        x = x.view(B, -1, 4 * C) ## B H/2*W/2 4C

        x = self.reduction(x)##线性变换，也即是全连接层，也是一层感知机
        x = self.norm(x)

        return x
##空间选择机制
class SSF(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.act = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.act1 = nn.ReLU(inplace=True)  
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):                
            nn.init.constant_(m.weight, 1.0)                
            nn.init.constant_(m.bias, 0)

 
    def forward(self, xc, x):
        B,C,H,W = x.shape
        attns = xc+x
        x_avg = torch.mean(x, dim=1, keepdim=True)
        xc_avg = torch.mean(xc, dim=1, keepdim=True)
        xnn= torch.abs(x_avg-xc_avg)
        # xnc = self.act1(x_avg-xc_avg)
        # x_1 = self.act(xc_avg-xnn)
        w1 = self.sig(xnn)
        w2 = torch.sigmoid(self.act(x_avg))
        w3 = torch.sigmoid(self.act(xc_avg))
        pool = torch.cat([w1, w2, w3], dim=1)
        max_pool = torch.max(pool, dim=1, keepdim=True)[0]
        attn1 = max_pool*attns

        x1= self.act1(xc-x)  
        x1 = self.act(x-x1)        
        v1 = F.softmax(x1.flatten(2),dim=2)
        v1 = v1.view(B,C,H,W)
        attn2 = v1*attns 
        attn = attns+attn1+attn2

        return attn
        

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:##DSC
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2,
        groups=out_planes//32,##out_planes,##out_planes
        bias=True,
        dilation=dilation,
    )

class Bottleneck(nn.Module):

    # 每个stage维度中扩展的倍数
    expansion: int = 4
    '''
          :param inplanes: 输入block的之前的通道数
          :param planes: 在block中间处理的时候的通道数
                  planes*self.extention:输出的维度
          :param stride:步长
          :param downsample:下采样
          '''
    def __init__(
        self,
        inplanes,
        planes,
        groups,
        dilation,
        norm_layer=nn.LayerNorm,##Callable用于表示可调用对象的类型。
        # 一个函数接受另一个函数作为参数，或者返回一个函数，可以使用Callable来指定类型
    ) -> None:
        super().__init__()
        drop = 0.0
        width = inplanes
        self.conv1 = conv5x5(inplanes, width, 1, groups, dilation)
        self.conv = conv1x1(width,width)
        dim = width
        self.norm1 = norm_layer(dim,eps=1e-6)
        self.conv2 = nn.Linear(dim, 4 * dim)##conv1x1(width, planes * self.expansion)  ##nn.Linear(dim, dim * self.expansion)##输入为dim，输出为dim*4
        self.act = nn.GELU()
        self.conv3 = nn.Linear(4 * dim, dim)##conv1x1( planes * self.expansion,width)
        self.norm2 = norm_layer(dim,eps=1e-6)
        self.drop = nn.Dropout(drop)
        
        
    def forward(self, x) :
        identity = x
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)# (N, C, H, W) -> (N, H, W, C)
        identity = identity.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = identity + x
        ##=========================================
        out = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.drop(x)+out
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
       
        return x
     
