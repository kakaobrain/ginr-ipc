from .configs import TransINRNerfConfig, LowRankModulatedTransINRNerfConfig
from .transinr_nerf import TransINRNerf
from .low_rank_modulated_transinr_nerf import LowRankModulatedTransINRNerf


def transinr_nerf(config: TransINRNerfConfig):
    return TransINRNerf(config)


def low_rank_modulated_transinr_nerf(config: LowRankModulatedTransINRNerfConfig):
    return LowRankModulatedTransINRNerf(config)
