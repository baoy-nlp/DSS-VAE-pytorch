# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

TINY = 1e-9

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


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


def unk_replace(input_sequence, dropoutr, vocab):
    if dropoutr > 0.:
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available(): prob = prob.cuda()
        prob[(input_sequence.data - vocab.sos_id) * (input_sequence.data - vocab.pad_id) * (
                input_sequence.data - vocab.eos_id) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < dropoutr] = vocab.unk_id
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

    seqs_t = []
    masks = []
    for i in range(max_len):
        seqs_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return seqs_t, masks


def input_padding(sents, pad_token, max_len):
    batch_size = len(sents)
    seqs_t = []
    masks = []
    for i in range(max_len):
        seqs_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])
    return seqs_t, masks


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


def to_input_variable(sequences, vocab, max_len=-1, cuda=False, training=True, append_boundary_sym=False,
                      batch_first=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if not isinstance(sequences[0], list):
        sequences = [sequences]

    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    if max_len != -1:
        seqs_t, masks = input_padding(word_ids, vocab['<pad>'], max_len)
    else:
        seqs_t, masks = input_transpose(word_ids, vocab['<pad>'])

    if not training:
        with torch.no_grad():
            seqs_var = Variable(torch.LongTensor(seqs_t), requires_grad=False)
    else:
        seqs_var = Variable(torch.LongTensor(seqs_t), requires_grad=False)
    if cuda:
        seqs_var = seqs_var.cuda()

    if batch_first:
        seqs_var = seqs_var.transpose(1, 0).contiguous()

    return seqs_var


def to_target_word(log_prob, vocab):
    _, word_ids = log_prob.sort(dim=-1, descending=True)
    word_ids = word_ids[:, :, 0].cpu().numpy().tolist()
    return [[[id2word(sents, vocab)], [-1]] for sents in word_ids]


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


def mask_scores(scores, beam_mask, EOS):
    """
    Mask scores of next step according to beam mask.
    Args:
        scores (torch.Tensor): Scores of next tokens with shape [batch_size, beam_size, vocab_size].
            Smaller should be better (usually negative log-probs).
        beam_mask (torch.Tensor): Mask of beam. 1.0 means not closed and vice verse. The shape is
            [batch_size, beam_size]

    Returns:
        Masked scores of next tokens.
    """
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[EOS] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + \
             torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def reflection(x, dim):
    return x


def gumbel_softmax(inputs, beta=0.5, tau=1.0):
    noise = inputs.data.new(*inputs.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return F.softmax((inputs + beta * Variable(noise)) / tau, dim=-1)


def positional_encodings_like(x, t=None, use_cuda=True):  # hope to be differentiable
    """
    Args:
        x: [batch_size,length,hidden] hidden dim mush a
        t:
        use_cuda:
    """
    if t is None:
        positions = torch.arange(0, x.size(-2))  # .expand(*x.size()[:2])
        if use_cuda:
            positions = positions.cuda()
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1)  # 0 2 4 6 ... (256)
    if use_cuda:
        channels = channels.cuda()
    channels = 1 / (10000 ** Variable(channels).float())

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings
