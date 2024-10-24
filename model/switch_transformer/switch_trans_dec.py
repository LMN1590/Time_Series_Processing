import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward

from .switch_trans_ff import SwitchFeedForward

class SwitchTransformerDecoderLayer(Module):
    """
    # Switch Transformer Block

    This is the same as [normal transformer block](../models.html#TransformerLayer)
    with handling extra outputs of switch feedforward module.
    """

    def __init__(
        self, *,
        d_model: int,
        
        nheads:int,
        
        n_experts:int = 4,
        capacity_factor:float = 1.0,
        d_ff:int =2048,
        
        dropout_prob: float
    ):
        """
        * `d_model` is the token embedding size
        * `attn` is the attention module
        * `feed_forward` is the feed forward module (which is the switching module in this case)
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(
            heads=nheads,
            d_model=d_model,
            dropout_prob=dropout_prob
        )
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.norm1 = nn.LayerNorm([d_model])
        
        self.enc_dec_attn = MultiHeadAttention(
            heads=nheads,
            d_model=d_model,
            dropout_prob=dropout_prob
        )
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.norm2 = nn.LayerNorm([d_model])
        
        self.feed_forward = SwitchFeedForward(
            d_model=d_model,
            capacity_factor=capacity_factor,
            drop_tokens=False,
            is_scale_prob=True,
            n_experts=n_experts,
            expert=FeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout_prob
            )
        )
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.norm3 = nn.LayerNorm([d_model])
        
        self.size = d_model

    def forward(
        self,
        dec:torch.Tensor,dec_mask:torch.Tensor,
        enc:torch.Tensor,enc_mask:torch.Tensor
    ):
        
        dec_self_attn = self.self_attn(query=dec, key=dec, value=dec, mask=dec_mask)
        dec = self.norm1(dec + self.dropout1(dec_self_attn))
        
        if enc is not None:
            dec_cross_attn = self.enc_dec_attn(query=dec,key=enc,value=enc,mask=enc_mask)
            dec = self.norm2(dec + self.dropout2(dec_cross_attn))
        
        dec_ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(dec)
        dec = self.norm3(dec + self.dropout3(dec_ff))
        
        return dec, counts, route_prob, n_dropped, route_prob_max
        