"""
---
title: Switch Transformer
summary: >
  This is an annotated implementation/tutorial a miniature version of Switch Transformer in PyTorch.
---

# Switch Transformer

This is a miniature [PyTorch](https://pytorch.org) implementation of the paper
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961).
Our implementation only has a few million parameters and doesn't do model parallel distributed training.
It does single GPU training, but we implement the concept of switching as described in the paper.

The Switch Transformer uses different parameters for each token by switching among parameters
based on the token.
Therefore, only a fraction of parameters are chosen for each token.
So you can have more parameters but less computational cost.

The switching happens at the Position-wise Feedforward network (FFN) of each transformer block.
Position-wise feedforward network consists of two sequentially fully connected layers.
In switch transformer we have multiple FFNs (multiple experts),
and we chose which one to use based on a router.
The output is a set of probabilities for picking a FFN,
and we pick the one with the highest probability and only evaluate that.
So essentially the computational cost is the same as having a single FFN.
In our implementation this doesn't parallelize well when you have many or large FFNs since it's all
happening on a single GPU.
In a distributed setup you would have each FFN (each very large) on a different device.

The paper introduces another loss term to balance load among the experts (FFNs) and
discusses dropping tokens when routing is not balanced.

Here's [the training code](experiment.html) and a notebook for training a switch transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/experiment.ipynb)
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.utils import clone_module_list

from .switch_trans_enc import SwitchTransformerEncoderLayer
from .switch_trans_dec import SwitchTransformerDecoderLayer

class SwitchTransformerEncoder(Module):
    def __init__(self, layer: SwitchTransformerEncoderLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each transformer layer
        stacked_encs, counts, route_prob, n_dropped, route_prob_max = [], [], [], [], []
        for layer in self.layers:
            x, f, p, n_d, p_max = layer(x=x, mask=mask)
            stacked_encs.append(x)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        return torch.stack(stacked_encs), torch.stack(counts), torch.stack(route_prob), n_dropped, torch.stack(route_prob_max)
    
class SwitchTransformerDecoder(Module):
    def __init__(self, layer: SwitchTransformerDecoderLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(
        self, 
        dec: torch.Tensor, dec_mask: torch.Tensor,
        encs:torch.Tensor, encs_mask:torch.Tensor
    ):
        # Run through each transformer layer
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for idx,layer in enumerate(self.layers):
            dec, f, p, n_d, p_max = layer(
                dec=dec,dec_mask=dec_mask,
                enc=encs[idx],enc_mask=encs_mask[idx]
            )
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        return dec, torch.stack(counts), torch.stack(route_prob), n_dropped, torch.stack(route_prob_max)
