from .trainer_stage_inr import Trainer as TrainerINR
from .trainer_stage_meta_inr import Trainer as TrainerMetaINR
from .trainer_stage_inr_nerf import Trainer as TrainerNerf

STAGE_INR_ARCH_TYPE = ["transinr", "low_rank_modulated_transinr"]
STAGE_META_INR_ARCH_TYPE = ["meta_film_inr", "meta_low_rank_modulated_inr"]
STAGE_INR_NERF_TYPE = ["transinr_nerf", "low_rank_modulated_transinr_nerf"]

def create_trainer(config):
    if config.arch.type in STAGE_INR_ARCH_TYPE:
        return TrainerINR
    elif config.arch.type in STAGE_META_INR_ARCH_TYPE:
        return TrainerMetaINR
    elif config.arch.type in STAGE_INR_NERF_TYPE:
        return TrainerNerf
    else:
        raise ValueError("architecture type not supported")
