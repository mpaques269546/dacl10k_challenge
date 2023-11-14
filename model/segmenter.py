import torch
import torch.nn as nn
import math
import os
from torchvision import transforms




class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        pred_size = 512):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pred_size = pred_size
        
   
    def forward(self, x ):
        (B,C,H0,W0) = x.shape
        coeff = self.pred_size / max([H0,W0])
        x = transforms.Resize(size= (int(coeff*H0), int(coeff*W0)))(x) # resize s.t. max_size = 512

        pad_H = abs( x.shape[-2] - 512 )
        pad_W = abs( x.shape[-1] - 512 )
        

        x = transforms.Pad(padding =( pad_W, pad_H, 0, 0 ) , fill=0)(x) # padding ( left, top, right and bottom) # pad to have (512,512)
        features = self.encoder.forward(x)
        [_, mask] = self.decoder.forward(features[:4] )
        mask = mask[:,:,pad_H:, pad_W:] # unpad
        mask = transforms.Resize(size = (H0,W0))(mask) #resize initial size
        return mask






'''
## segment with multiple inferences for each crop (512,512)

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        pred_size = 512,
        max_img_size = 1024
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pred_size = pred_size
        self.max_img_size = max_img_size
      
        
    def forward_patch(self, patches):
        features = self.encoder.forward(patches)
        masks = self.decoder.forward(features[:4] )
        return masks
    
    def patchify(self, x, p=512):
        """
        x: (B, C, H, W)
        return: (B*n, C, p, p), n=H*W/p**2
        """
        assert x.dim()==4
        [B,C,H,W] = x.shape
        assert  H % p == 0 and W % p == 0
        h = H // p
        w = W // p
        n = h*w
        x = x.reshape(shape=(B, C, h, p, w, p))
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(shape=(B*n, C, p, p))
        return x, h, w

    def unpatchify(self, x, B, h, w,  p=512):
        """
        x: [B*n, C, p, p]
        return: (B, C, H, W)
        """
        [Bn, C, p, p] = x.shape
        #n = int(Bn/B)
        #w = h = int(n**.5)
        x = x.reshape(shape=(B, h, w, C, p, p))
        x = torch.einsum('bhwcpq->bchpwq', x)
        x = x.reshape(shape=(B, C, h * p, w * p))
        return x

    
    def forward(self, inp ):
        B0_, C0_, H0_, W0_  = inp.shape
        
        if H0_>self.max_img_size or W0_>self.max_img_size:
            inp = transforms.Resize( (self.max_img_size, self.max_img_size))(inp)
            B0, C0, H0, W0  = inp.shape
        else:
            B0, C0, H0, W0 = B0_, C0_, H0_, W0_
            
        pad_H =   (H0//self.pred_size + 1 ) * self.pred_size -  H0
        pad_W = (W0//self.pred_size +  1) * self.pred_size - W0
        x = F.pad(inp , ( 0, pad_W, 0, pad_H ) , "constant", 0)
        x, h, w  = self.patchify( x, p = self.pred_size )
        x = self.forward_patch(x)
        x = self.unpatchify(x, B0, h, w,  p = self.pred_size )
        x =  x[:,:,:H0, :W0]
        
        if H0_!=H0 or W0_!=W0:
            x = transforms.Resize( (H0_,W0_))(x)
        
        return x

'''