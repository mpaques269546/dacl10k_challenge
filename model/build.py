import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from torchvision import transforms

from .encoder import BEiTAdapter
from .decoder import UperNet, Mask2FormerHead


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if checkpoint_key!='' and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

            #print('pretrained=', state_dict.keys())
            #print('model sate dict=', model.state_dict().keys())

            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # remove `model.` prefix induced by saving models
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            
            # in case different inp size: change pos_embed 
            if "pos_embed" in state_dict.keys():
                pretrained_shape = state_dict['pos_embed'].size()[1]
                model_shape = model.state_dict()['pos_embed'].size()[1]
                if pretrained_shape != model_shape:
                    pos_embed = state_dict['pos_embed']
                    pos_embed = pos_embed.permute(0, 2, 1)
                    pos_embed = F.interpolate(pos_embed, size=model_shape)
                    pos_embed = pos_embed.permute(0, 2, 1)
                    state_dict['pos_embed'] = pos_embed
            if "cls_token" in state_dict.keys():
                pretrained_shape = state_dict['cls_token'].size()[1]
                model_shape = model.state_dict()['cls_token'].size()[1]
                if pretrained_shape != model_shape:
                    print('cls_token sizes did not match')
                    del state_dict['cls_token']
                else:
                    print('cls_token sizes matched')
           
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

        else:
            print('wrong checkpoint key')
            print("There is no reference weights available for this model => We use random weights.")
    else:
        print('file {0} does not exist'.format(pretrained_weights))

def build_model( pretrained_weights='', key='',trainable=False, arch=None, patch_size=8 , image_size=224,  with_cls_token=True, num_cls=2, activation=nn.Sigmoid(), embed_dim=384, num_queries=10):
    
    print(f'==== MODEL {arch} ====')
    
    if arch=='beit_large_adapter':
        model = BEiTAdapter( pretrain_size=640, conv_inplane=64, n_points=4, deform_num_heads=16, # pretrain_size=512
                 init_values=0., cffn_ratio=0.25, deform_ratio=0.5, with_cffn=True,
                 interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]], add_vit_feature=True, with_cp=False,
                 # vit params
                 patch_size=16,   embed_dim=1024, depth=24,
                 num_heads=16, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.3, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False,)
    elif arch=='upernet':
            #model = UperNet(num_cls=num_cls, embed_dim=embed_dim, hidden_dim =embed_dim, activation=activation, freeze_bn=False ) # freeze_bn = False
            hidden_size = 1024 # cocostuff # 512 # ade20k
            model = UperNet( pool_scales=(1, 2, 3, 6), in_channels=embed_dim, hidden_size = hidden_size, num_labels= num_cls, activation = activation, upsample_after=True )
    elif arch == 'mask2former':
        model = Mask2FormerHead(in_channels=embed_dim,strides=[4, 8, 16, 32],num_pixel_layer=6,num_transformer_layer=9, feat_channels=embed_dim[0],feedforward_channels=4096, 
            out_channels=embed_dim[0], num_classes= num_cls, num_queries=num_queries, num_heads=32, num_transformer_feat_level=3) #  num_queries=num_cls
        
    else:
        print(f'ERROR wrong architecture {arch}')

    # load weights to evaluate
    load_pretrained_weights(model, pretrained_weights, key, arch)
        
    # freeze or not weights
    ct, cf =0 ,0
    if trainable:
        for p in model.parameters():
            p.requires_grad = True
            ct+= p.numel()
    else:
        for p in model.parameters():
            p.requires_grad = False
            cf+= p.numel()
    print(f"Parameters: {ct} trainable, {cf} frozen.")
    return model









        