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

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.attentive_encoder import SelfATTEncoder
from nn_self.mapper import MatrixMapper
from utils.nn_funcs import positional_encodings_like


def find_val(inputs, val, axis=1):
    """
    Args:
        inputs: batch,max_len
        val: eos id
        axis: dim
    Return:
        is_find: byteTensor [batch,1]
        indices: longTensor [batch,1]
    """
    val_match = (inputs == val)
    return ((val_match.cumsum(axis) == 1) & val_match).max(axis)


class MatrixDecoder(nn.Module):
    """
    Input:
        z: batch_size, hidden
    Constructing:
        Inv(R): max_len X max_len
        S:  max_len X hidden_dim
    Modules:
        Matrix Mapper: z -> Inv(R)
        Matrix Mapper: z -> S
        Predictor: Inv(R)*S -> Tgt_V
    """

    def __init__(self, vocab_size, max_len, input_dim, hidden_dim, mapper_dropout=0.1, out_dropout=0.1, pad_id=0,
                 **kwargs):
        super(MatrixDecoder, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.control_matrix_mapper = MatrixMapper(input_dim, hidden_dim, self.max_len, self.max_len, mapper_dropout)
        self.semantic_matrix_mapper = MatrixMapper(input_dim, hidden_dim, self.max_len, hidden_dim, mapper_dropout)
        self.word_predictor = nn.Sequential(
            # nn.Dropout(out_dropout),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Dropout(out_dropout),
            nn.Linear(hidden_dim, vocab_size, bias=True)
        )
        self.pad_id = pad_id

    def get_semantic(self, sem_inputs):
        return self.semantic_matrix_mapper.forward(sem_inputs)

    def generate(self, con_inputs, sem_inputs):
        # con_mat = self.control_matrix_mapper.forward(con_inputs)
        sem_mat = self.get_semantic(sem_inputs)
        # dec_init = torch.bmm(con_mat, sem_mat)
        # return self.word_predictor.forward(dec_init)
        return self.word_predictor.forward(sem_mat)

    def forward(self, inputs, **kwargs):
        return self.generate(inputs, inputs)

    def score(self, raw_score, tgt_var, **kwargs):
        """
        Args:
            raw_score: [seq_len, batch_size, vocab_size]
            tgt_var: [batch_size, seq_len] or [seq_len, batch_size]
        """
        scores = self.forward(raw_score, **kwargs)
        scores = scores.contiguous().transpose(1, 0).contiguous()
        batch_size = scores.size(1)

        if tgt_var.size(0) == batch_size:
            tgt_var = tgt_var.contiguous().transpose(1, 0)
        log_probs = F.log_softmax(scores.view(-1, self.vocab_size).contiguous(), dim=-1)
        flattened_tgt_var = tgt_var.contiguous().view(-1)  # [seq_len * batch_size]

        sent_log_probs = torch.gather(log_probs, 1, flattened_tgt_var.unsqueeze(1)).squeeze(1)
        sent_log_probs = sent_log_probs * (1. - torch.eq(flattened_tgt_var, self.pad_id).float())  # 0 is pad
        sent_log_probs = sent_log_probs.view(-1, batch_size).sum(dim=0)  # [batch_size]

        return sent_log_probs


class MatrixAttDecoder(MatrixDecoder):
    def __init__(self,
                 vocab_size,
                 max_len,
                 input_dim,
                 hidden_dim,
                 n_layers,
                 n_head,
                 inner_dim,
                 block_dropout,
                 dim_per_head,
                 mapper_dropout=0.1,
                 out_dropout=0.1,
                 pad_id=0,
                 use_cuda=True,
                 ):
        super(MatrixAttDecoder, self).__init__(
            vocab_size,
            max_len,
            input_dim,
            hidden_dim,
            mapper_dropout,
            out_dropout,
            pad_id
        )

        self.self_att_encoder = SelfATTEncoder(
            n_layers,
            hidden_dim,
            inner_dim,
            n_head,
            block_dropout,
            dim_per_head,
        )
        self.use_cuda = use_cuda

    def generate(self, con_inputs, sem_inputs):
        next_sem_inputs = self.get_semantic(sem_inputs)
        next_sem_pos = positional_encodings_like(next_sem_inputs, use_cuda=self.use_cuda)
        sem_mat = self.self_att_encoder.forward(out=next_sem_inputs + next_sem_pos)
        return self.word_predictor.forward(sem_mat)
