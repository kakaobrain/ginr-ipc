from .ema import ExponentialMovingAverage
from .generalizable_INR import transinr, low_rank_modulated_transinr, meta_low_rank_modulated_inr
from .generalizable_INR_nerf import transinr_nerf, low_rank_modulated_transinr_nerf


def create_model(config, ema=False):
    model_type = config.type.lower()
    if model_type == "transinr":
        model = transinr(config)
        model_ema = transinr(config) if ema else None
    elif model_type == "transinr_nerf":
        model = transinr_nerf(config)
        model_ema = transinr_nerf(config) if ema else None
    elif model_type == "low_rank_modulated_transinr":
        model = low_rank_modulated_transinr(config)
        model_ema = low_rank_modulated_transinr(config) if ema else None
    elif model_type == "low_rank_modulated_transinr_nerf":
        model = low_rank_modulated_transinr_nerf(config)
        model_ema = low_rank_modulated_transinr_nerf(config) if ema else None
    elif model_type == "meta_low_rank_modulated_inr":
        model = meta_low_rank_modulated_inr(config)
        model_ema = meta_low_rank_modulated_inr(config) if ema else None
    else:
        raise ValueError(f"{model_type} is invalid..")

    if ema:
        model_ema = ExponentialMovingAverage(model_ema, config.ema)
        model_ema.eval()
        model_ema.update(model, step=-1)

    return model, model_ema
