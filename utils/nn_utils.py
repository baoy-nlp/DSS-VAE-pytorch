# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def bag_of_word_loss(input_log_softmax, input_var, criterion):
    """
    :param input_log_softmax: [batch_size,vocab_size]
    :param input_var: [batch_size,max_step]
    :param criterion
    :return:
    """
    seq_length = input_var.size(1)
    batch_size = input_var.size(0)
    vocab_size = input_log_softmax.size(-1)
    origin_score = input_log_softmax.view(batch_size, 1, vocab_size)
    expand_log_score = origin_score.expand((batch_size, seq_length, vocab_size)).contiguous().view(-1, vocab_size)
    return criterion(expand_log_score, input_var.view(-1))


def noise_input(input_sequence, droprate, vocab):
    if droprate > 0.:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available(): prob = prob.cuda()
        prob[(input_sequence.data - vocab.sos_id) * (input_sequence.data - vocab.pad_id) * (input_sequence.data - vocab.eos_id) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < droprate] = vocab.unk_id
        return decoder_input_sequence
    return input_sequence


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == "fixed":
        return 1.0
    elif anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'sigmoid':
        return float(1 / (1 + np.exp(0.001 * (x0 - step))))
    elif anneal_function == 'negative-sigmoid':
        return float(1 / (1 + np.exp(-0.001 * (x0 - step))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def wd_anneal_function(unk_max, anneal_function, step, k, x0):
    return unk_max * kl_anneal_function(anneal_function, step, k, x0)


def word_dropout(input_sequence, droprate, vocab):
    if droprate > 1e-6:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available(): prob = prob.cuda()
        prob[(input_sequence.data - vocab.sos_id) * (input_sequence.data - vocab.pad_id) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < droprate] = vocab.unk_id
        return decoder_input_sequence
    return input_sequence


def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


def length_array_to_mask_tensor(length_array, cuda=False):
    max_len = length_array[0]
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [robust_id2word(s, vocab) for s in sents]
    else:
        return robust_id2word(sents, vocab)


def robust_id2word(sents, vocab):
    res = []
    for w in sents:
        if w == vocab.sos_id or w == vocab.pad_id:
            pass
        elif w == vocab.eos_id:
            break
        else:
            res.append(vocab.id2word[w])
    return res


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def to_input_variable(sequences, vocab, cuda=False, training=True, append_boundary_sym=False, batch_first=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if not isinstance(sequences[0], list):
        sequences = [sequences]

    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    if not training:
        with torch.no_grad():
            sents_var = Variable(torch.LongTensor(sents_t), requires_grad=False)
    else:
        sents_var = Variable(torch.LongTensor(sents_t), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    if batch_first:
        sents_var = sents_var.transpose(1, 0).contiguous()

    return sents_var


def variable_constr(x, v, cuda=False):
    return Variable(torch.cuda.x(v)) if cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()
