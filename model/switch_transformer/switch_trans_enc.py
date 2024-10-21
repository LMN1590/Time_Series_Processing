import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward

from .switch_trans_ff import SwitchFeedForward

class SwitchTransformerEncoderLayer(Module):
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
        self.size = d_model
        self.attn = MultiHeadAttention(
            heads=nheads,
            d_model=d_model,dropout_prob=dropout_prob
        )
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
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        self.norm_final = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the switching feed-forward network
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        # Add the feed-forward results back
        x = self.norm_final(x + self.dropout(ff))

        return x, counts, route_prob, n_dropped, route_prob_max
