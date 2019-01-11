import torch.nn as nn

from submodules.base_decoder import RNNDecoder
from submodules.base_encoder import RNNEncoder
from submodules.beam_decoder import TopKDecoder
from submodules.bridge import Bridge
from utils.nn_utils import *


class BaseSeq2seq(nn.Module):
    def forward(self, **kwargs):
        pass

    def __init__(self, args, src_vocab, tgt_vocab, src_embed=None, tgt_embed=None):
        super(BaseSeq2seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.args = args
        self.encoder = RNNEncoder(
            vocab=len(src_vocab),
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=src_embed
        )

        self.enc_factor = 2 if args.bidirectional else 1
        self.enc_dim = args.enc_hidden_dim * self.enc_factor

        if args.mapper_type == "link":
            self.dec_hidden = self.enc_dim
        elif args.use_attention:
            self.dec_hidden = self.enc_dim
        else:
            self.dec_hidden = args.dec_hidden_dim

        self.bridger = Bridge(
            rnn_type=args.rnn_type,
            mapper_type=args.mapper_type,
            encoder_dim=self.enc_dim,
            encoder_layer=args.enc_num_layers,
            decoder_dim=self.dec_hidden,
            decoder_layer=args.dec_num_layers,
        )

        self.decoder = RNNDecoder(
            vocab=len(tgt_vocab),
            max_len=args.tgt_max_time_step,
            input_size=args.dec_embed_dim,
            hidden_size=self.dec_hidden,
            embed_droprate=args.dec_ed,
            rnn_droprate=args.dec_rd,
            n_layers=args.dec_num_layers,
            rnn_cell=args.rnn_type,
            use_attention=args.use_attention,
            embedding=tgt_embed,
            eos_id=tgt_vocab.eos_id,
            sos_id=tgt_vocab.sos_id,
        )

        self.beam_decoder = TopKDecoder(
            decoder_rnn=self.decoder,
            k=args.sample_size
        )
        print("seq-to-seq {} layers {} with attention: {}".format(args.num_layers, args.rnn_type, args.use_attention))

    def init(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def encode(self, src_var, src_length):
        encoder_outputs, encoder_hidden = self.encoder.forward(input_var=src_var, input_lengths=src_length)
        return encoder_outputs, encoder_hidden

    def bridge(self, encoder_hidden):
        # batch_size = encoder_hidden.size(1)
        # convert = encoder_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bridger.forward(encoder_hidden)

    def score(self, examples, return_enc_state=False):
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
            tgt_words = [e.tgt for e in examples]
        else:
            src_words = examples.src
            tgt_words = examples.tgt

        src_var = to_input_variable(src_words, self.src_vocab, cuda=args.cuda, batch_first=True)
        tgt_var = to_input_variable(tgt_words, self.tgt_vocab, cuda=args.cuda, append_boundary_sym=True, batch_first=True)
        src_length = [len(c) for c in src_words]

        encoder_outputs, encoder_hidden = self.encode(src_var=src_var, src_length=src_length)
        encoder_hidden = self.bridge(encoder_hidden)
        scores = self.decoder.score(inputs=tgt_var, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs)

        if return_enc_state:
            return scores, encoder_hidden
        else:
            return scores

    def beam_search(self, src_sent, beam_size=5, dmts=None):
        if dmts is None:
            dmts = self.args.decode_max_time_step
        src_var = to_input_variable(src_sent, self.src_vocab,
                                    cuda=self.args.cuda, training=False, append_boundary_sym=False, batch_first=True)
        src_length = [len(src_sent)]

        encoder_outputs, encoder_hidden = self.encode(input_var=src_var, input_lengths=src_length)
        encoder_hidden = self.bridger.forward(input_tensor=encoder_hidden)
        meta_data = self.beam_decoder.beam_search(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            beam_size=beam_size,
            decode_max_time_step=dmts
        )
        topk_sequence = meta_data['sequence']
        topk_score = meta_data['score'].squeeze()

        completed_hypotheses = torch.cat(topk_sequence, dim=-1)

        number_return = completed_hypotheses.size(0)
        final_result = []
        final_scores = []
        for i in range(number_return):
            hyp = completed_hypotheses[i, :].data.tolist()
            res = id2word(hyp, self.tgt_vocab)
            final_result.append(res)
            final_scores.append(topk_score[i].item())
        return final_result, final_scores
