import torch.nn as nn

from decoder.attentive_decoder import TransformerDecoder
from encoder.attentive_encoder import TransformerEncoder
from nn_self.criterions import SequenceCriterion
from utils.nest import map_structure
from utils.nn_funcs import to_input_variable
from utils.tensor_ops import tensor_gather_helper
from utils.tensor_ops import tile_batch


class Transformer(nn.Module):
    """
    A sequence to sequence model with attention mechanism.
    """

    # def __init__(
    #         self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
    #         d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
    #         dropout=0.1, proj_share_weight=True, **kwargs):
    def __init__(self, args, src_vocab, tgt_vocab):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad = 0
        n_src_vocab = len(self.src_vocab)
        n_tgt_vocab = len(self.tgt_vocab)
        d_word_vec = args.enc_embed_dim
        d_model = args.enc_hidden_dim

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        self.encoder = TransformerEncoder(
            vocab_size=n_src_vocab,
            n_layers=args.enc_num_layers,
            n_head=args.enc_head,
            input_size=d_word_vec,
            hidden_size=d_model,
            inner_hidden=args.enc_inner_hidden,
            embed_dropout=args.enc_ed,
            block_dropout=args.enc_rd,
            dim_per_head=None,
            pad=self.pad
        )

        self.decoder = TransformerDecoder(
            vocab_size=n_tgt_vocab,
            n_layers=args.dec_num_layers,
            n_head=args.dec_head,
            input_size=d_word_vec,
            hidden_size=d_model,
            inner_hidden=args.dec_inner_hidden,
            dim_per_head=None,
            share_proj_weight=args.share_proj_weight,
            share_embed_weight=self.encoder.embeddings if args.share_embed_weight else None,
            embed_dropout=args.dec_ed,
            block_dropout=args.dec_rd,
            pad=self.pad
        )

        self.dropout = nn.Dropout(args.drop)
        self.generator = self.decoder.generator

        self.normalization = 1.0,
        self.norm_by_words = False
        self.critic = SequenceCriterion(padding_idx=self.pad)

    def forward(self, src_seq, tgt_seq, log_probs=True):

        ret = self.encode(src_seq)
        enc_output = ret['ctx']
        enc_mask = ret['ctx_mask']
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)

        return self.generator(dec_output, log_probs=log_probs)

    def encode(self, src_seq):

        enc_out = self.encoder(src_seq)

        return {"ctx": enc_out['out'], "ctx_mask": enc_out['mask']}

    def init(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(
            tgt_seq=tgt_seq,
            enc_output=ctx,
            enc_mask=ctx_mask,
            enc_attn_caches=enc_attn_caches,
            self_attn_caches=slf_attn_caches
        )

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)
        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states

    def score(self, examples, return_enc_state=False):
        """
            Used for teacher-forcing training,
            return the log_probability of <input,output>.
        """
        args = self.args
        if isinstance(examples, list):
            src_words = [e.src for e in examples]
            tgt_words = [e.tgt for e in examples]
        else:
            src_words = examples.src
            tgt_words = examples.tgt

        seqs_x = to_input_variable(src_words, self.src_vocab, cuda=args.cuda, batch_first=True)
        seqs_y = to_input_variable(tgt_words, self.tgt_vocab, cuda=args.cuda, append_boundary_sym=True,
                                   batch_first=True)
        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        words_norm = y_label.ne(self.pad).float().sum(1)

        log_probs = self.forward(seqs_x, y_inp)
        loss = self.critic(inputs=log_probs, labels=y_label, reduce=False, normalization=self.normalization)

        if self.norm_by_words:
            loss = loss.div(words_norm).sum()
        else:
            loss = loss.sum()
        return loss
