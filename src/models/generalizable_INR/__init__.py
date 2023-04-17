from .transinr import TransINR
from .low_rank_modulated_transinr import LowRankModulatedTransINR
from .meta_low_rank_modulated_inr import MetaLowRankModulatedINR

def transinr(config):
    return TransINR(config)

def low_rank_modulated_transinr(config):
    return LowRankModulatedTransINR(config)

def meta_low_rank_modulated_inr(config):
    return MetaLowRankModulatedINR(config)
