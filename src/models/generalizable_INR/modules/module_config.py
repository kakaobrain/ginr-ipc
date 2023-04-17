from typing import List, Optional, Any
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING


@dataclass
class Unfold1DConfig:
    patch_size: int = 200
    use_padding: bool = True


@dataclass
class UnfoldConfig:
    patch_size: int = 9
    padding: int = 1


@dataclass
class VQGANEncoderConfig:
    model_path: Optional[str] = None
    model_name: Optional[str] = "cae16"
    quantize: bool = False


@dataclass
class CLIPImageEncoderConfig:
    clip_name: str = "ViT-B/16"
    apply_ln_post: bool = True
    encoding_token_type: str = "all"  # "cls" | "spatial-only" | "all"


@dataclass
class DataEncoderConfig:
    type: str = "unfold"
    n_channel: int = 3
    trainable: bool = False
    encoder_spec: Any = None

    def __post_init__(self):
        supported_types = {
            "unfold": UnfoldConfig,
            "unfold_audio": Unfold1DConfig,
            "vqgan": VQGANEncoderConfig,
            "clip": CLIPImageEncoderConfig,
        }

        try:
            config_cls = supported_types[self.type]
        except KeyError:
            raise ValueError(f"unsupported DataEncoder type {self.type} (must be in {supported_types.keys()})")

        default_spec = OmegaConf.structured(config_cls())
        if self.encoder_spec is None:
            self.encoder_spec = default_spec
        self.encoder_spec = OmegaConf.merge(default_spec, self.encoder_spec)


@dataclass
class LatentMappingConfig:
    type: str = "linear"
    n_patches: int = 400
    n_layer: int = 1
    activation: str = "relu"
    hidden_dim: List[int] = MISSING
    latent_dim: int = 256
    use_pe: bool = True


@dataclass
class FourierMappingConfig:
    type: Optional[str] = "deterministic_transinr"
    trainable: bool = False
    use_ff: bool = True
    ff_sigma: int = 1024
    ff_dim: int = 128


@dataclass
class HypoNetActivationConfig:
    type: str = "relu"
    siren_w0: Optional[float] = 30.0


@dataclass
class HypoNetInitConfig:
    weight_init_type: Optional[str] = "kaiming_uniform"
    bias_init_type: Optional[str] = "zero"


@dataclass
class HypoNetConfig:
    type: str = "mlp"
    n_layer: int = 5
    hidden_dim: List[int] = MISSING
    use_bias: bool = True
    input_dim: int = 2
    output_dim: int = 3
    output_bias: float = 0.5
    fourier_mapping: FourierMappingConfig = FourierMappingConfig()
    activation: HypoNetActivationConfig = HypoNetActivationConfig()
    initialization: HypoNetInitConfig = HypoNetInitConfig()

    normalize_weight: bool = True


@dataclass
class FiLMHypoNetConfig:
    type: str = "mlp"
    n_layer: int = 5
    hidden_dim: List[int] = MISSING
    use_bias: bool = True
    input_dim: int = 2
    output_dim: int = 3
    output_bias: float = 0.5
    rescale_film: bool = False
    fourier_mapping: FourierMappingConfig = FourierMappingConfig()
    activation: HypoNetActivationConfig = HypoNetActivationConfig()
    initialization: HypoNetInitConfig = HypoNetInitConfig()


@dataclass
class CoordSamplerConfig:
    data_type: str = "image"
    coord_range: List[float] = MISSING
    train_strategy: Optional[str] = MISSING
    val_strategy: Optional[str] = MISSING
