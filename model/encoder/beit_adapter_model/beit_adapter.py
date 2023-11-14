# Copyright (c) Shanghai AI Lab. All rights reserved.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .beit_transformer import BEiTBaseline
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, InteractionBlockWithCls
from .ms_deform_attn import MSDeformAttn
        
class BEiTAdapter(BEiTBaseline):
    def __init__(self, pretrain_size=224, conv_inplane=64, n_points=4, deform_num_heads=16,
                 init_values=0., cffn_ratio=0.25, deform_ratio=0.5, with_cffn=True,
                 interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]], add_vit_feature=True, with_cp=False,
                 # vit params
                 patch_size=16,   embed_dim=1024, depth=24,
                 num_heads=16, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.3, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 *args, **kwargs):

        super().__init__(img_size=pretrain_size, patch_size=patch_size,embed_dim=embed_dim, depth=depth,num_heads=num_heads,
                         qkv_bias=qkv_bias,qk_scale=qk_scale, use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,drop_path_rate=drop_path_rate,
                         out_indices=None,  init_values=init_values, with_cp=with_cp, *args, **kwargs)

        # self.num_classes = 80
        # self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.feat_dim = [embed_dim for i in range(4)]
       

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=True if i == len(interaction_indexes) - 1 else False,
                             with_cp=with_cp,  with_cls_token=True)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm1 = nn.SyncBatchNorm(embed_dim) if self.device=='cuda' else nn.Identity()
        self.norm2 = nn.SyncBatchNorm(embed_dim) if self.device=='cuda' else nn.Identity()
        self.norm3 = nn.SyncBatchNorm(embed_dim) if self.device=='cuda' else nn.Identity()
        self.norm4 = nn.SyncBatchNorm(embed_dim) if self.device=='cuda' else nn.Identity()

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        #self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        
        # rel pos bias
        #rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, (H, W) = self.patch_embed(x)
        bs, n, dim = x.shape
    
        # add the [CLS] token to the embed patch tokens
        cls_token = self.cls_token.expand(bs, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if self.patch_size==16:
            H_, W_ = 2*H, 2*W
        if self.patch_size==8:
            H_, W_ = H, W
            
        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed, H, W)
            x = x + pos_embed
        x = self.pos_drop(x)
        

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                              deform_inputs1, deform_inputs2, H_, W_) # is rel_pos_bias, rel_pos_bias = rel_pos_bias)
            cls_token, patch_token = x[:, :1, ],  x[:, 1:, ] # remove cls token
            outs.append(patch_token.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_ , W_ ).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_//2 , W_//2 ).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_ // 4, W_ // 4).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            if self.patch_size==16:
                x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
                x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False) #x3: same dim
                x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            if self.patch_size==8:
                x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)#x2 = same dim
                x3 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x4, scale_factor=0.25, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4, cls_token]