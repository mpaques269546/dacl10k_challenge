# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample

from .utils import caffe2_xavier_init, SinePositionalEncoding
from .transformer_decoder import Mask2FormerTransformerDecoder
from .pixel_decoder import MSDeformAttnPixelDecoder


# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
#from mmengine.model import ModuleList, caffe2_xavier_init
#from mmengine.structures import InstanceData
from torch import Tensor

#from mmdet.registry import MODELS, TASK_UTILS
#from mmdet.structures import SampleList
#from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
#from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
#from ..utils import get_uncertain_point_coords_with_randomness
#from .anchor_free_head import AnchorFreeHead
#from .maskformer_head import MaskFormerHead



class Mask2FormerHead(nn.Module):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int]=[1024,1024,1024,1024],
                 strides:List[int]=[4, 8, 16, 32],
                 num_pixel_layer:int=6,
                 num_transformer_layer:int=9,
                 feat_channels: int=1024,
                 feedforward_channels:int=4096,
                 out_channels: int=1024,
                 num_classes: int = 19,
                 num_queries: int = 40,
                 num_heads:int=32,
                 num_transformer_feat_level: int = 3,
                 enforce_decoder_input_project: bool = True,
                 **kwargs) -> None:
        super().__init__()
        positional_encoding = dict(num_feats= feat_channels//2 , normalize=True) # feat_channels//2
        self.pixel_decoder = MSDeformAttnPixelDecoder(num_layers=num_pixel_layer, num_encoder_levels=num_transformer_feat_level, in_channels=in_channels, strides=strides, feat_channels=feat_channels,out_channels= out_channels, 
                                                      num_outs = 3, norm_cfg = dict(type='GN', num_groups=32),
                 act_cfg = dict(type='ReLU'), positional_encoding = positional_encoding, self_attn_cfg= dict(embed_dims=feat_channels, num_heads=num_heads, num_levels=num_transformer_feat_level, dropout=0.0), 
                 ffn_cfg = dict(embed_dims=feat_channels, feedforward_channels=feedforward_channels, num_fcs=2,  ffn_drop=0.,act_cfg=dict(type='ReLU', inplace=True)), )
        
        self.transformer_decoder = Mask2FormerTransformerDecoder(num_layers=num_transformer_layer, self_attn_cfg= dict(embed_dims=feat_channels, num_heads=num_heads, dropout=0.0),
                                       cross_attn_cfg= dict(embed_dims=feat_channels,  num_heads=num_heads, dropout=0.0),
                                       ffn_cfg = dict(embed_dims=feat_channels, feedforward_channels=feedforward_channels, num_fcs=2,  ffn_drop=0.,act_cfg=dict(type='ReLU', inplace=True)),
                                       norm_cfg = dict(type='LN'))
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = num_heads  #self.transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = self.transformer_decoder.num_layers
        #assert self.pixel_decoder.encoder.layer_cfg. \ self_attn_cfg.num_levels == num_transformer_feat_level
        
       
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs =nn.ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)
        #print('self.level_embed', self.level_embed.weight.shape )

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.init_weights()


    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: list,
                             batch_img_metas: list) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask
    
    
    def forward(self, features):
        out = self.forward_train(features)
        cls_score, mask_pred = out['pred_logits'] , out['pred_masks']

        # semantic inference
        #cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        #mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='bilinear')
        #seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred).sigmoid()
        
        # semantic inference (meta)
        cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_mask = torch.einsum("bqc,bqhw->bchw", cls_score, mask_pred)
        seg_mask = F.interpolate(seg_mask , scale_factor=4, mode='bilinear').clamp(min=0, max=1)
        return [out, seg_mask ]

    def forward_train(self, x: List[Tensor]) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor
        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size, d, h, w = x[0].shape
        mask_features, multi_scale_memorys = self.pixel_decoder(x )
        # mask_features torch.Size([b,d,h/4,w/4]) multi_scale_memorys [[b,d,h/4,w/4],[[b,d,h/8,w/8],[b,d,h/16,w16]]
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        # dict {'pred_logits', 'pred_masks', 'aux_outputs'}
        # in list 'aux_outputs' : 'pred_logits' 'pred_masks'
        output = {'pred_logits': cls_pred_list[-1], 'pred_masks': mask_pred_list[-1], 'aux_outputs': [{'pred_logits': cls_pred_list[i] , 'pred_masks': mask_pred_list[i]} for i in range(len(cls_pred_list) -1 )] }
        #return cls_pred_list, mask_pred_list
        return output