# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn

from nn_self.embeddings import Embeddings
from nn_self.sublayers import PositionwiseFeedForward, MultiHeadedAttention


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class SelfATTEncoder(nn.Module):
    def __init__(self, n_layers, hidden_size, inner_hidden, n_head, block_dropout, dim_per_head=None):
        super().__init__()
        self.num_layers = n_layers
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=hidden_size, d_inner_hid=inner_hidden, n_head=n_head, dropout=block_dropout,
                          dim_per_head=dim_per_head)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, out, enc_slf_attn_mask=None):
        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out


class TransformerEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_layers=6,
                 n_head=8,
                 input_size=512,
                 hidden_size=512,
                 inner_hidden=1024,
                 embed_dropout=0.1,
                 block_dropout=0.1,
                 dim_per_head=None,
                 pad=0,
                 **kwargs
                 ):
        super().__init__()
        self.pad_id = pad
        self.embeddings = Embeddings(num_embeddings=vocab_size,
                                     embedding_dim=input_size,
                                     dropout=embed_dropout,
                                     add_position_embedding=True,
                                     padding_idx=self.pad_id
                                     )

        self.self_att_encoder = SelfATTEncoder(
            n_layers,
            hidden_size,
            inner_hidden,
            n_head,
            block_dropout,
            dim_per_head
        )
        self.hiddne_size = hidden_size

    def reset_embed(self, share_embed):
        self.embeddings = share_embed

    @property
    def out_dim(self):
        return self.hiddne_size

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(self.pad_id)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = self.self_att_encoder(emb, enc_slf_attn_mask)

        # return out, enc_mask
        return {
            "out": out,
            "mask": enc_mask,
        }
