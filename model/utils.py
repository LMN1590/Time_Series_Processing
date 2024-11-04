import torch.nn as nn
import torch

from timm.layers import trunc_normal_
import math

from typing import Tuple

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        size:Tuple[int],
        random:bool,
        pos_embed_require_grad:bool
    ):
        super(PositionalEncoding, self).__init__()       
        
        if(random):
            self.pos_embed = self._generate_random_embedding(size)
        else:
            self.pos_embed = self._generate_definitive_embedding(size)
        
        self.pos_embed = nn.Parameter(self.pos_embed)
        self.pos_embed.requires_grad = pos_embed_require_grad
        

    def forward(self, x):
        return x + self.pos_embed
    
    def _generate_random_embedding(
        self,
        size:Tuple[int]
    ):
        embed = torch.zeros(size)
        trunc_normal_(embed,std=.02)
        return embed
    
    def _generate_definitive_embedding(
        self,
        size:Tuple[int]
    ):
        assert len(size)==3, "Non-random embeddings only support tensor with shape (T,B,I)."
        T,B,I = size
        
        embed = torch.zeros(T,I)
        position = torch.arange(0,T,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, I, 2).float() * (-math.log(10000.0) / I))
        embed[:, 0::2] = torch.sin(position * div_term)
        embed[:, 1::2] = torch.cos(position * div_term)
        
        embed = embed.unsqueeze(1)
        
        return embed