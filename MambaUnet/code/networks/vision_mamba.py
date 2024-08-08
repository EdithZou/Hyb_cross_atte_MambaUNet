# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM, VSSM_attn

logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    def __init__(self, config, in_cha=3, img_size=96, num_classes=21843, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.in_chs = in_cha
        self.mamba_unet =  VSSM(
                                patch_size=config.MODEL.VSSM.PATCH_SIZE,
                                in_chans=self.in_chs,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.VSSM.EMBED_DIM,
                                depths=config.MODEL.VSSM.DEPTHS,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    
    def load_from(self, config):
        #pretrained_path = config.MODEL.PRETRAIN_CKPT
        pretrained_path = '/home/tzou/MyoSAIQ/Mamba_UNet/code/pretrained_ckpt/vmamba_tiny_e292.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            #device = torch.device('cpu')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)

        else:
            print("none pretrain")

class Segmentation_Block(nn.Module):
    def __init__(self, config, in_chans, num_classes):
        super(Segmentation_Block, self).__init__()
        self.mamba_unet = VSSM_attn(
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=config.MODEL.VSSM.EMBED_DIM,
            depths=config.MODEL.VSSM.DEPTHS,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )
        self.load_pretrained_model(config)
        
    def load_pretrained_model(self, config):
        #pretrained_path = config.MODEL.PRETRAIN_CKPT
        pretrained_path = '/home/tzou/MyoSAIQ/Mamba_UNet/code/pretrained_ckpt/vmamba_tiny_e292.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)

        else:
            print("none pretrain")
                
    def forward(self, x):
        return self.mamba_unet(x)
            
class CrossAttentionMambaUnet(nn.Module):
    def __init__(self, config, in_cha=3, img_size=96, num_classes=21843, zero_head=False, vis=False):
        super(CrossAttentionMambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.in_chs = in_cha
        self.seg_block = nn.ModuleList([Segmentation_Block(
                                        config, in_chans=self.in_chs, num_classes=2)
                                        for i in range(self.num_classes)
                                        ])
        # 改为多头子网络
        self.num_heads = self.num_classes
        self.attn_feats = 16
        # Create separate heads for each class
        self.heads = nn.ModuleList([nn.Linear(config.MODEL.VSSM.EMBED_DIM, 1) for _ in range(self.num_heads)])
        self.conv =  nn.Conv2d(in_channels=config.MODEL.VSSM.EMBED_DIM, out_channels=self.attn_feats, kernel_size=1, bias=False)
        self.attn_map = CrossAttention_Map(image_size=96, attn_features=self.attn_feats, numbers=self.num_classes)
        self.conv_edge=Conv_edge(self.attn_feats*2)
        self.seg_edge=nn.Conv2d(self.attn_feats,2 * self.num_classes ,kernel_size=3, padding=3//2)        
        self.conv_dice=Conv_dice(self.attn_feats*2)
        self.seg_dice=nn.Conv2d(self.attn_feats,self.num_classes,kernel_size=3,padding=3//2)        
        self.conv_merge = Conv_merge(in_channels=32)
        self.up = FinalPatchExpand_X4(dim_scale=4,dim=4)
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_ls = []
        f_ls =[]
        pred_ls =[]
        for i in range(self.num_classes):
            x_, f_, pred_ = self.seg_block[i](x)
            x_, f_ = self.conv(x_), self.conv(f_)
            x_ls.append(x_)
            f_ls.append(f_)
            pred_ls.append(pred_)
        attn_x = self.attn_map(x_ls, f_ls)
        f_dice = self.conv_dice(attn_x)
        f_edge = self.conv_edge(attn_x)
        pred_edge = self.seg_edge(f_edge)
        
        pred_dice = self.seg_dice(self.conv_merge(torch.cat([f_dice,f_edge],dim=1))+f_dice)
        pred_dice = self.up(pred_dice)
        pred_edge = self.up(pred_edge)
        pred_ls = torch.cat(pred_ls, dim=1)
        return pred_ls, pred_dice, pred_edge
    


class CrossAttention_Map(nn.Module):
    def __init__(self,image_size,attn_features,numbers) -> None:
        super().__init__()
        self.numbers=numbers

        self.attn_map=nn.ModuleDict({f"{i}_{j}":CrossAttention(image_size//4 * image_size//4, attn_features).cuda() for i in range(numbers) for j in range(numbers)})
        self.conv_re_ls=nn.ModuleList([nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1),nn.Conv2d(attn_features*numbers,16,kernel_size=3,padding=3//2)).cuda() for _ in range(numbers)])
        self.conv_cat=nn.Conv2d(16*numbers,16*2,kernel_size=3,padding=3//2)    
    def forward(self,x_ls,f_x_ls):
        attn_x_parts = []
        for i in range(self.numbers):
            inner_attn_parts = []
            for j in range(self.numbers):
                attn_result = self.attn_map[f"{i}_{j}"](f_x_ls[i], f_x_ls[j])
                inner_attn_parts.append(attn_result)
            inner_cat_result = torch.cat(inner_attn_parts, dim=1)
            conv_result = self.conv_re_ls[i](inner_cat_result) + x_ls[i]
            attn_x_parts.append(conv_result)
        attn_x=torch.cat(attn_x_parts, dim=1)          
        #attn_x=torch.cat([self.conv_re_ls[i](torch.cat([(seclf.attn_map[f"{i}_{j}"](f_x_ls[i],f_x_ls[j])) for j in range(self.numbers)],dim=1))+x_ls[i] for i in range(self.numbers)],dim=1)
        attn_x=self.conv_cat(attn_x)

        return attn_x  #channels=32

class CrossAttention(nn.Module):
    def __init__(self, dim,in_channels, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.conv_cat=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        self.conv_chanl = nn.Conv2d(96, 16, kernel_size=1)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x1,x2):
        B, N, H, W = x1.shape
        C=H*W
        # print(x1.shape)
        x_cat=torch.cat([x1,x2],dim=1)
        x_cat=self.conv_cat(x_cat).reshape(B,N,C)


        q = self.wq(x1.reshape(B,N,C)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = self.wk(x_cat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x_cat).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BHNN @ BHN(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x_attn = self.proj(x_attn).reshape(B,N,H,W)+x1

        # x = self.proj_drop(x)
        return x_attn


class Conv_edge(nn.Sequential):
    def __init__(self,in_channels) -> None:
        # CONV: conv+bn+relu
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)

class Conv_dice(nn.Sequential):
    def __init__(self,in_channels) -> None:
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)

class Conv_merge(nn.Sequential):
    def __init__(self,in_channels) -> None:
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)



class Conv_merge(nn.Sequential):
    def __init__(self,in_channels) -> None:
        conv1=nn.Conv2d(in_channels,in_channels*2,kernel_size=3,padding=3//2)
        bn1=nn.BatchNorm2d(in_channels*2)
        # ln1=nn.LayerNorm([in_channels*2,224,224])
        relu1= nn.ReLU(inplace=True)
        conv2=nn.Conv2d(in_channels*2,in_channels,kernel_size=3,padding=3//2)
        bn2=nn.BatchNorm2d(in_channels)
        # ln2=nn.LayerNorm([in_channels,224,224])
        relu2=nn.ReLU(inplace=True)
        conv3=nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=3//2)
        bn3=nn.BatchNorm2d(in_channels//2)
        # ln3=nn.LayerNorm([in_channels//2,224,224])
        relu3=nn.ReLU(inplace=True)
        super().__init__(conv1,bn1,relu1,conv2,bn2,relu2,conv3,bn3,relu3)

from einops import rearrange
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.dim_scale, mode='bilinear', align_corners=False)
        #x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        #self.attn_map = 
    def forward(self, x):
        # x: (batch_size, num_classes, height, width, hidden_dim)
        B, num_classes, H, W, C = x.shape
        x = x.view(B, num_classes, -1).permute(2, 0, 1)  # (H*W, B, num_classes)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 2, 0).view(B, num_classes, H, W)
    

