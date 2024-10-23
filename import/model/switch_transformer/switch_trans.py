import torch.nn as nn
import torch

from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention

from einops import repeat

from model.utils import PositionalEncoding

from .switch_trans_enc import SwitchTransformerEncoderLayer
from .switch_trans_dec import SwitchTransformerDecoderLayer
from .switch_trans_blocks import (
    SwitchTransformerEncoder,
    SwitchTransformerDecoder
)

class SwitchTransformer(nn.Module):
    def __init__(
        self,
        input_dim:int = 50,
        in_len:int = 16,
        output_dim:int = 10,
        
        random_embed:bool = True,
        pos_embed_requires_grad:bool = True,
        
        d_model:int = 512,
        d_ff:int = 2048,
        nheads:int = 8,
        dropout_prob:float = .1,
        n_layers:int = 4,
        
        n_experts:int = 4,
        capacity_factor:float = 1.0
    ):
        super(SwitchTransformer, self).__init__()
        
        self.preprocess = nn.Linear(input_dim,d_model)
        
        self.pos_embed = PositionalEncoding(
            size=(in_len,1,d_model),
            random=random_embed,
            pos_embed_require_grad=pos_embed_requires_grad
        )
        
        self.switch_encoder = SwitchTransformerEncoder(
            SwitchTransformerEncoderLayer(
                d_model=d_model,
                nheads=nheads,
                n_experts=n_experts,
                capacity_factor=capacity_factor,
                d_ff=d_ff,
                dropout_prob=dropout_prob
            ),
            n_layers=n_layers
        )
        
        self.switch_decoder = SwitchTransformerDecoder(
            SwitchTransformerDecoderLayer(
                d_model=d_model,
                nheads=nheads,
                n_experts=n_experts,
                capacity_factor=capacity_factor,
                d_ff=d_ff,
                dropout_prob=dropout_prob
            ),
            n_layers=n_layers
        )
        
        self.decoder_fluff = nn.Parameter(
            torch.randn((1,1,d_model))
        )
        
        self.final_output = nn.Linear(d_model,output_dim)
        
        self.n_layers = n_layers
        self._init_weights()
    
    def forward(
        self, x:torch.Tensor
    ):
        B,T,I = x.shape
        
        x = self.preprocess(x)
        x = x.permute(1,0,2)
        x = self.pos_embed(x)
        encs, counts_enc, route_prob_enc, n_dropped_enc, route_prob_max_enc = self.switch_encoder(
            x,torch.ones(T, T, 1).float().cuda()
        )
        
        decoder_seed = repeat(self.decoder_fluff, 'ts b d -> ts (repeat b) d',repeat = B)
        x, counts_dec, route_prob_dec, n_dropped_dec, route_prob_max_dec= self.switch_decoder(
            decoder_seed, torch.triu(torch.ones(1, 1, 1)).float().cuda(),
            encs, torch.ones(self.n_layers,1,T,1).float().cuda()
        )
        x = x.permute(1,0,2).flatten(1)
        x = self.final_output(x)
        return x
    
    def _init_weights(self):
        initrange = 0.1    
        self.final_output.bias.data.zero_()
        self.final_output.weight.data.uniform_(-initrange, initrange)