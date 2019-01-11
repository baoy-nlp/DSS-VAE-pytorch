from syntaxVAE.ae import AutoEncoder
from syntaxVAE.disentangle_vae import DisentangleVAE
from syntaxVAE.evaluation import evaluate
from syntaxVAE.evaluation import evaluate_autoencoder
from syntaxVAE.evaluation import evaluate_parser
from syntaxVAE.evaluation import evaluate_reconstructor
from syntaxVAE.evaluation import evaluate_vae
from syntaxVAE.syntax_guide_vae import SyntaxGuideVAE
from syntaxVAE.syntax_vae import SyntaxVAE
from syntaxVAE.vanilla_vae import VanillaVAE

MODEL_CLS = {
    'AutoEncoder': AutoEncoder,
    'OriginVAE': VanillaVAE,
    'FakeCoupleVAE': SyntaxGuideVAE,
    'DVAE': DisentangleVAE,
    'SVAE': SyntaxVAE,
}


def build_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)
