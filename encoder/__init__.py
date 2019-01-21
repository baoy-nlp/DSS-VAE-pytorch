from .attentive_encoder import TransformerEncoder
from .rnn_encoder import RNNEncoder

MODEL_CLS = {
    'rnn': RNNEncoder,
    'att': TransformerEncoder,
}


def get_encoder(model: str, **kwargs):
    if model.lower() not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    args = kwargs['args']
    return MODEL_CLS[model.lower()](
            vocab_size=kwargs["vocab_size"],
            max_len=args.src_max_time_step,
            input_size=args.enc_embed_dim,
            hidden_size=args.enc_hidden_dim,
            embed_droprate=args.enc_ed,
            rnn_droprate=args.enc_rd,
            n_layers=args.enc_num_layers,
            bidirectional=args.bidirectional,
            rnn_cell=args.rnn_type,
            variable_lengths=True,
            embedding=kwargs['embed'],
            n_head=args.enc_head,
            inner_hidden=args.enc_inner_hidden,
            embed_dropout=args.enc_ed,
            block_dropout=args.enc_rd,
            dim_per_head=None,
            pad=kwargs['pad']
        )
