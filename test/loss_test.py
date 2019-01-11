import torch
import torch.nn.functional as F


def bow_loss(input_log_softmax, input_var, criterion):
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


def base_examples():
    score = torch.Tensor([1, 2] * 6).view(2, 1, 6)
    score.require_grad = True
    prob = F.log_softmax(score, dim=-1)

    labels = torch.Tensor(
        [0, 1, 2, 3]
    ).long().view(2, 2)

    seq_len = labels.size(-1)
    size = torch.Tensor(2, seq_len, 6)
    wp_scores = prob.expand(size.size()).contiguous()
    print(wp_scores)
    nll = torch.nn.NLLLoss(ignore_index=1)
    output = nll(wp_scores.view(-1, 6), labels.view(-1))
    print(output)


score = torch.Tensor([1, 2] * 6).view(2, 6)
score.require_grad = True

input_score = F.log_softmax(score)
labels = torch.Tensor(
    [0, 1, 2, 3]
).long().view(2, 2)
criterion = torch.nn.NLLLoss(ignore_index=1)

print(bow_loss(input_score, labels, criterion))

base_examples()
