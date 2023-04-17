from typing import List, Optional
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING
from .modules.module_config import (
    DataEncoderConfig,
    LatentMappingConfig,
    HypoNetConfig,
    CoordSamplerConfig,
)


@dataclass
class AttentionBlockConfig:
    embed_dim: int = MISSING
    n_head: int = MISSING
    mlp_bias: bool = True
    attn_bias: bool = True
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    gelu: str = "v1"


@dataclass
class AttentionStackConfig:
    n_layer: int = MISSING
    embed_dim: int = 768
    mask: bool = False
    block: AttentionBlockConfig = AttentionBlockConfig()


@dataclass
class TransformerConfig:
    n_layer: int = MISSING
    embed_dim: int = 768
    mask: bool = False
    block: AttentionBlockConfig = AttentionBlockConfig()


@dataclass
class TransINRConfig:
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
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        default_dataenc_config = DataEncoderConfig(type=config.data_encoder.type)
        defaults = OmegaConf.structured(cls(ema=False, data_encoder=default_dataenc_config))
        config = OmegaConf.merge(defaults, config)
        config.transformer.block.embed_dim = config.transformer.embed_dim
        return config


@dataclass
class MappingNetConfig:
    hidden_dim: int = 64
    n_layer: int = 1


@dataclass
class LowRankModulatedTransINRConfig:
    type: str = "low_rank_modulated_transinr"
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


@dataclass
class MetaLowRankModulatedINRConfig:
    type: str = "meta_low_rank_modulated_inr"
    ema: Optional[bool] = None

    n_inner_step: int = 2
    inner_lr: float = 0.01

    hyponet: HypoNetConfig = HypoNetConfig()

    rank: List[int] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None
    use_factorization: bool = True

    coord_sampler: CoordSamplerConfig = CoordSamplerConfig()

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config
