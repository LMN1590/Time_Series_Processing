import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .cross_encoder import Encoder
from .cross_decoder import Decoder
from .attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from .cross_embed import DSW_embedding
from model.utils import PositionalEncoding

from math import ceil

class Crossformer(nn.Module):
    def __init__(
        self, 
        data_dim, in_len, out_len, seg_len, output_dim,
        random_embed:bool = True,pos_embed_requires_grad:bool=True,
        win_size = 4, factor=10, d_model=512,
        d_ff = 1024, n_heads=8, e_layers=3, 
        dropout=0.0, baseline = False
    ):
        super(Crossformer, self).__init__()
        assert random_embed, "Non-random positional embedding is not supported"
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.output_dim = output_dim

        self.baseline = baseline


        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.pos_embed = PositionalEncoding(
            size=(1, data_dim, (self.pad_in_len // seg_len), d_model),
            random = random_embed,
            pos_embed_require_grad=pos_embed_requires_grad
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
        self.fc = nn.Linear(
            self.data_dim*self.out_len, self.output_dim
        )
        
    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq = self.pos_embed(x_seq)
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        decoded_prediction = base + predict_y[:, :self.out_len, :]
        decoded_prediction = decoded_prediction.flatten(1)
        result = self.fc(decoded_prediction)
        return result