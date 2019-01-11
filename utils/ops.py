import torch


def get_tensor(val):
    x = torch.Tensor(val)
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_long_tensor(val):
    return get_tensor(val).long()


def get_byte_tensor(val):
    return get_tensor(val).byte()


def get_float_tensor(val):
    return get_tensor(val).float()


def inflate(tensor, times, dim):
    """
    Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)

    Args:
        tensor: A :class:`Tensor` to inflate
        times: number of repetitions
        dim: axis for inflation (default=0)

    Returns:
        A :class:`Tensor`

    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> b = ._inflate(a, 2, dim=1)
        >> b
        1   2   1   2
        3   4   3   4
        [torch.LongTensor of size 2x4]
        >> c = _inflate(a, 2, dim=0)
        >> c
        1   2
        3   4
        1   2
        3   4
        [torch.LongTensor of size 4x2]

    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


def sequence_mask(sequence_length, max_len: int):
    batch_size = sequence_length.size(0)
    sequence_length = sequence_length.view(-1, 1)
    seq_range = get_tensor(range(0, max_len)).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.expand_as(seq_range_expand))
    return seq_range_expand.le(seq_length_expand).long()


def batch_slice_set(input, dim, index, to_set):
    index = index.squeeze()
    _input = input * 1
    if dim == 1:
        _input[range(index.size()[0]), index, :] = to_set
    else:
        _input[range(index.size()[0]), :, index] = to_set
    return _input


def batch_slice_select(input, dim, index):
    """
    def batch_slice_select(input,dim,index):
        index = index.squeeze()
        if dim == 1:
            return input[range(index.size()[0]), index, :]
        else:
            return input[range(index.size()[0]), :, index]
    Returns a new tensor which indexes the input 3-D tensor along with the batch-dimension and the dim using index which is a LongTensor
        :param input(Tensor): the input 3-D Tensor
        :param dim(int): the dimension in which we index
        :param index(LongTensor): the 1-D tensor containing the indices to index
        :return: torch.Tensor


    Examples::

        >>>import torch
        >>>x=torch.Tensor(range(36)).view(2,3,6)
        tensor([[[  0.,   1.,   2.,   3.,   4.,   5.],
                 [  6.,   7.,   8.,   9.,  10.,  11.],
                 [ 12.,  13.,  14.,  15.,  16.,  17.]],

                [[ 18.,  19.,  20.,  21.,  22.,  23.],
                 [ 24.,  25.,  26.,  27.,  28.,  29.],
                 [ 30.,  31.,  32.,  33.,  34.,  35.]]])
        >>>ind=torch.Tensor([1,0]).long()
        >>>batch_slice_select(input,0,ind)
        tensor([[  1.,   7.,  13.],
                [ 18.,  24.,  30.]])
        >>>batch_slice_select(input,1,ind)
        tensor([[  6.,   7.,   8.,   9.,  10.,  11.],
                [ 18.,  19.,  20.,  21.,  22.,  23.]])
    """
    index = index.squeeze()
    if dim == 1:
        return input[range(index.size()[0]), index, :]
    else:
        return input[range(index.size()[0]), :, index]


def batch_elements_select(input, index):
    index = index.squeeze()
    return input[range(index.size()[0]), index]


def reflect(input, **kwargs):
    return input


def zero_initialize(layers, batch_size, hidden_dims, rnn_cell='lstm'):
    initial_val = [0.0] * (layers * batch_size * hidden_dims)
    if rnn_cell == 'lstm':
        return [
            get_tensor(initial_val).view(layers, batch_size, hidden_dims),
            get_tensor(initial_val).view(layers, batch_size, hidden_dims),
        ]
    else:
        get_tensor(initial_val).view(layers, batch_size, hidden_dims)


def rnn_initialize(encoder_hidden, bidirectional_encoder):
    def _cat_directions(h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    if encoder_hidden is None:
        return None
    if isinstance(encoder_hidden, tuple):
        encoder_hidden = tuple([_cat_directions(h) for h in encoder_hidden])
    else:
        encoder_hidden = _cat_directions(encoder_hidden)
    return encoder_hidden
