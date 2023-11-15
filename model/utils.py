import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

def masked_image(pil_image, mask ):
    
    class_list =  ['Background', 'Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity',
                   'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 
                   'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']
    num_cls = len(class_list)
    palette = torch.tensor([[1,0,0],
        [0,1,0],
        [0,0,1],
        [1,0.5,0.5],
        [0.5,1,0.5],
        [0.5,0.5,1],
        [1,1,0.5],
        [1,0.5,1],
        [0.5,1,1],
        [1,1,0.25],
        [1,0.25,1],
        [0.25,1,1],
        [0.5,0,0],
        [0,0.5,0],
        [0,0,0.5],
        [0.25,0,0],
        [0,0.25,0],
        [0,0,0.25],
        [0.75,0,0],
        [0,0.75,0],
        [0,0,0.75],

        ])
    mask = (mask.squeeze()>0.5)*1.
    mask = torch.cat( (torch.zeros_like(mask[0]).unsqueeze(0),mask), dim=0) # add background class
    mask = mask.argmax(0) # consider multiclass monolabel pb to simplify plotting
    cls_pred = mask.unique()
    img = transforms.ToTensor()(pil_image )
    masked_im = img.clone()

    for n in range(1,num_cls+1):
        color = palette[n]
        pred = (mask==n)*1
        for c in range(3):
            masked_im[c] = masked_im[c]*(1-pred) + pred*color[c] *masked_im[c]
    label_pred = [class_list[i] for i in cls_pred[1:]]
    return transforms.ToPILImage()(masked_im.float()), label_pred


def make_prediction(segmenter, pil_image ):
    transform =  transforms.Compose([ transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    mask = segmenter(transform(pil_image).unsqueeze(0))
    masked_im, label_pred =  masked_image(pil_image, mask.squeeze() )
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text('original image')
    plt.imshow(pil_image)
    plt.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    
    ax.title.set_text('pred:'+str(label_pred))
    plt.imshow(masked_im)
    plt.axis('off')
