from .attentive_decoder import TransformerDecoder
from .parallel_decoder import MatrixAttDecoder
from .parallel_decoder import MatrixDecoder
from .rnn_decoder import RNNDecoder

MODEL_CLS = {
    'rnn': RNNDecoder,
    'mat': MatrixDecoder,
    'mat-att': MatrixAttDecoder,
}


def get_decoder(model: str, **kwargs):
    if model.lower() not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))
    args = kwargs['args']
    return MODEL_CLS[model.lower()](
        vocab_size=kwargs["vocab_size"],
        max_len=args.tgt_max_time_step,
        input_dim=kwargs["input_dim"],
        hidden_dim=args.dec_hidden_dim,
        n_layers=args.dec_num_layers,
        n_head=args.dec_head,
        inner_dim=args.dec_inner_hidden,
        block_dropout=args.dec_rd,
        mapper_dropout=args.dropm,
        out_dropout=args.dropo,
        dim_per_head=None,
        pad_id=kwargs['pad'],
        use_cuda=args.cuda,
    )
