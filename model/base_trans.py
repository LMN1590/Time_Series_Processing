import math

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import repeat

from .utils import PositionalEncoding 


class BaseTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim:int,
        time_steps_len:int,
        
        random_embed:bool,
        pos_embed_requires_grad:bool,
        
        d_model:int,
        nhead:int,
        num_layers:int,
        dropout_prob:float,
        output_dim:int
    ):
        super(BaseTransformerModel, self).__init__()
        
        self.preprocess = nn.Linear(input_dim,d_model)
        print(self.preprocess.weight.dtype)
        
        self.pos_embed = PositionalEncoding(
            size = (time_steps_len,1,d_model),
            random=random_embed,
            pos_embed_require_grad=pos_embed_requires_grad
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout_prob
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout_prob
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=num_layers+1)
        self.decoder_fluff = nn.Parameter(
            torch.randn((1,1,d_model))
        )
        
        self.mean_output = nn.Linear(d_model,output_dim)
        self.std_output = nn.Linear(d_model,output_dim)
        self.std_act = nn.Softplus()
        self._init_weights()
        
    def forward(self,x:torch.Tensor):
        B,T,I = x.shape
        x = self.preprocess(x)
        x = x.permute(1,0,2)
        x = self.pos_embed(x)
        x = self.encoder(x)
        
        decoder_seed = repeat(self.decoder_fluff, 'ts b d -> ts (repeat b) d',repeat = B)
        x = self.decoder(decoder_seed,x)
        x = x.permute(1,0,2)
        x = x.flatten(1)
        mean = self.mean_output(x).flatten()
        std = self.std_act(self.std_output(x)).flatten()
        return torch.stack([mean,std])
        
    def _init_weights(self):
        initrange = 0.1    
        self.mean_output.bias.data.zero_()
        self.mean_output.weight.data.uniform_(-initrange, initrange)
        
        self.std_output.bias.data.zero_()
        self.std_output.weight.data.uniform_(-initrange, initrange)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask