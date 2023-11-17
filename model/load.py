import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from torchvision import transforms


from .segmenter import Segmenter
from .build import build_model



def load_segmenter(pretrained_weights):
	num_cls = 19
	img_size = 512
	activation = torch.nn.Identity()
	encoder = build_model(pretrained_weights,  key='encoder', trainable=False, embed_dim=1024, 
	                                arch='beit_large_adapter', patch_size=16,  image_size=img_size)
	decoder = build_model(pretrained_weights, key='decoder', arch='mask2former', trainable=False, 
	 num_cls=num_cls, embed_dim=encoder.feat_dim, image_size=img_size, activation=activation, num_queries=10)
	segmenter = Segmenter( encoder, decoder, pred_size = img_size )
	return segmenter.eval()

