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


class MatrixMapper(nn.Module):
    """
    Input:
        z: batch_size,hidden
    Modules:
        mapper_k: z->hidden->k
        mapper_v: z->hidden->v
        k * v
    """

    def __init__(self, input_dim, hidden_dim, k_dim, v_dim, dropout=0.1):
        super(MatrixMapper, self).__init__()
        self.k_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k_dim, bias=True)
        )
        self.v_mapper = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, v_dim, bias=True),
        )

    def forward(self, inputs):
        """
        Mapper the inputs to a matrix
        Args:
            inputs:(Tensor: batch_size, hidden) encoder output or latent variable.
        """
        k_vec = self.k_mapper.forward(inputs)
        v_vec = self.v_mapper.forward(inputs)
        batch_size = inputs.size(0)
        post_k = k_vec.contiguous().view(batch_size, -1, 1)
        post_v = v_vec.contiguous().view(batch_size, 1, -1)

        return torch.bmm(post_k, post_v)
