from models.ae import AutoEncoder
from models.disentangle_vae import DisentangleVAE
from models.parallel_ae import ParallelAE
from models.syntax_guide_vae import SyntaxGuideVAE
from models.syntax_vae import SyntaxVAE
from models.vanilla_vae import VanillaVAE

MODEL_CLS = {
    'AutoEncoder': AutoEncoder,
    'VanillaVAE': VanillaVAE,
    'SyntaxGuideVAE': SyntaxGuideVAE,
    'DisentangleVAE': DisentangleVAE,
    'SyntaxVAE': SyntaxVAE,
    'ParallelAE': ParallelAE,
}


def init_create_model(model: str, **kwargs):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model](**kwargs)


def load_static_model(model: str, model_path: str):
    if model not in MODEL_CLS:
        raise ValueError(
            "Invalid model class \'{}\' provided. Only {} are supported now.".format(
                model, list(MODEL_CLS.keys())))

    return MODEL_CLS[model].load(model_path)
