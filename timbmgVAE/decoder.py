import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from timbmgVAE.encoder import to_var


class SentenceDecoder(nn.Module):
    """
    params:
    vocab: [vocab.src[
    args: contains
        embed_size
        word_dropout
        embedding_dropout
        rnn_type
        hidden_size
    """

    def __init__(self, args, vocab, word_embed=None):
        super(SentenceDecoder, self).__init__()
        self.args = args
        self.vocab = vocab
        self.bidirectional = args.bidirectional
        self.num_layers = args.num_layers
        self.sos_idx = vocab.sos_id
        self.eos_idx = vocab.eos_id
        self.pad_idx = vocab.pad_id
        self.unk_idx = vocab.unk_id
        self.latent_size = args.latent_size
        self.hidden_size = args.hidden_size
        self.word_dropout_rate = args.word_dropout

        if word_embed is None:
            self.embedding = nn.Embedding(len(vocab), args.embed_size)
        else:
            self.embedding = word_embed

        self.embedding_dropout = nn.Dropout(p=args.embedding_dropout)

        if args.rnn_type == 'rnn':
            rnn = nn.RNN
        elif args.rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()
        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers
        self.decoder_rnn = rnn(args.embed_size, args.hidden_size, num_layers=self.num_layers, dropout=args.dropout, bidirectional=self.bidirectional,
                               batch_first=True)
        self.latent2hidden = nn.Linear(self.latent_size, args.hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(args.hidden_size * (2 if self.bidirectional else 1), len(vocab))

    def forward(self, input_sequence, length, z):
        batch_size = input_sequence.size(0)
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()  # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self.sample(logits)

            # save next input
            generations = self.save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    @staticmethod
    def sample(dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    @staticmethod
    def save_sample(save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
        }
        torch.save(params, path)
