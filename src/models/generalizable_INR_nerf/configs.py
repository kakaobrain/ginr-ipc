from typing import List, Optional
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING
from .modules.module_config import CoordSamplerConfig
from ..generalizable_INR.modules.module_config import (
    DataEncoderConfig,
    LatentMappingConfig,
    HypoNetConfig,
)
from ..generalizable_INR.configs import TransformerConfig


@dataclass
class TransINRNerfConfig:
    type: str = "transinr"
    ema: Optional[bool] = None

    data_encoder: DataEncoderConfig = DataEncoderConfig()  # type, trainable, ckpt
    latent_mapping: LatentMappingConfig = LatentMappingConfig()  # type, n_layer, activation, hidden_dim, latent_dim
    transformer: TransformerConfig = TransformerConfig()
    hyponet: HypoNetConfig = HypoNetConfig()

    coord_sampler: CoordSamplerConfig = CoordSamplerConfig()
    n_weight_groups: List[int] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None

    @classmethod
    def create(cls, config):
        default_dataenc_config = DataEncoderConfig(type=config.data_encoder.type)
        defaults = OmegaConf.structured(cls(ema=False, data_encoder=default_dataenc_config))
        config = OmegaConf.merge(defaults, config)
        config.transformer.block.embed_dim = config.transformer.embed_dim
        return config


@dataclass
class LowRankModulatedTransINRNerfConfig:
    type: str = "low_rank_modulated_transinr_nerf"
    ema: Optional[bool] = None

    data_encoder: DataEncoderConfig = DataEncoderConfig()  # type, trainable, ckpt
    latent_mapping: LatentMappingConfig = LatentMappingConfig()  # type, n_layer, activation, hidden_dim, latent_dim
    transformer: TransformerConfig = TransformerConfig()
    hyponet: HypoNetConfig = HypoNetConfig()

    coord_sampler: CoordSamplerConfig = CoordSamplerConfig()

    n_weight_groups: List[int] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None

    @classmethod
    def create(cls, config):
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        default_dataenc_config = DataEncoderConfig(type=config.data_encoder.type)
        defaults = OmegaConf.structured(cls(ema=False, data_encoder=default_dataenc_config))
        config = OmegaConf.merge(defaults, config)
        config.transformer.block.embed_dim = config.transformer.embed_dim
        return config
