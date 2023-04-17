import numpy as np
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..configs import HypoNetConfig
from .utils import create_params_with_init, create_activation


class HypoNet(nn.Module):
    def __init__(self, config: HypoNetConfig):
        """
        It is equivalanet to the hyponet for images.
        The only difference is additional implementation of fourier mapping for nerf.
        """
        super().__init__()
        self.config = config
        self.use_bias = config.use_bias
        self.ff_config = config.fourier_mapping
        self.init_config = config.initialization
        self.use_ff = self.ff_config.use_ff
        self.num_layer = config.n_layer
        self.hidden_dims = config.hidden_dim
        if len(self.hidden_dims) == 1:
            self.hidden_dims = OmegaConf.to_object(self.hidden_dims) * (self.num_layer - 1)  # exclude output layer
        else:
            assert len(self.hidden_dims) == self.num_layer - 1

        if self.config.activation.type == "siren":
            assert not self.ff_config.use_ff
            assert self.init_config.weight_init_type == "siren"
            assert self.init_config.bias_init_type == "siren"

        if self.use_ff:
            assert self.ff_config.type is not None
            self.setup_fourier_mapping(ff_type=self.ff_config.type, trainable=self.ff_config.trainable)

        # after computes the shape of trainable parameters, initialize them
        self.params_dict = None
        self.params_shape_dict = self.compute_params_shape()
        self.activation = create_activation(self.config.activation)
        self.build_base_params_dict(self.config.initialization)
        self.output_bias = config.output_bias

        self.normalize_weight = config.normalize_weight
        self.ignore_base_param_dict = {name: False for name in self.params_dict}

    def compute_params_shape(self):
        """
        Computes the shape of MLP parameters.
        The computed shapes are used to build the initial weights by `build_base_params_dict`. 
        """
        config, ff_config = self.config, self.ff_config
        use_bias = self.use_bias

        param_shape_dict = dict()

        if not ff_config.use_ff:
            fan_in = config.input_dim
        else:
            fan_in = ff_config.ff_dim * 2

        fan_in = fan_in + 1 if use_bias else fan_in
        for i in range(config.n_layer - 1):
            fan_out = self.hidden_dims[i]
            param_shape_dict[f"linear_wb{i}"] = (fan_in, fan_out)
            fan_in = fan_out + 1 if use_bias else fan_out

        param_shape_dict[f"linear_wb{config.n_layer-1}"] = (fan_in, config.output_dim)
        return param_shape_dict

    def build_base_params_dict(self, init_config):
        assert self.params_shape_dict
        params_dict = nn.ParameterDict()
        for idx, (name, shape) in enumerate(self.params_shape_dict.items()):
            is_first = idx == 0
            params = create_params_with_init(
                shape,
                init_type=init_config.weight_init_type,
                include_bias=self.use_bias,
                bias_init_type=init_config.bias_init_type,
                is_first=is_first,
                siren_w0=self.config.activation.siren_w0,  # valid only for siren
            )
            params = nn.Parameter(params)
            params_dict[name] = params
        self.set_params_dict(params_dict)

    def check_valid_param_keys(self, params_dict):
        predefined_params_keys = self.params_shape_dict.keys()
        for param_key in params_dict.keys():
            if param_key in predefined_params_keys:
                continue
            else:
                raise KeyError

    def set_params_dict(self, params_dict):
        self.check_valid_param_keys(params_dict)
        self.params_dict = params_dict

    def setup_fourier_mapping(self, ff_type, trainable=False):
        """
        build the linear mapping for converting coordinates into fourier features
        """
        ff_sigma, ff_dim = self.ff_config.ff_sigma, self.ff_config.ff_dim
        if ff_type == "deterministic_transinr":
            log_freqs = torch.linspace(0, np.log(ff_sigma), ff_dim // self.config.input_dim)
            self.ff_linear = torch.exp(log_freqs)
        elif ff_type == "random_gaussian":
            self.ff_linear = torch.randn(self.config.input_dim, ff_dim) * ff_sigma  # scaler
        elif ff_type == "deterministic_transinr_nerf":
            self.ff_linear = 2 ** torch.linspace(0, ff_sigma, ff_dim // self.config.input_dim)
        else:
            raise NotImplementedError

        self.ff_linear = nn.Parameter(self.ff_linear, requires_grad=trainable)

    def fourier_mapping(self, coord):
        """
        Computes the fourier features of each coordinate based on configs.

        Args
            coord (torch.Tensor) : `coord.shape == (B, -1, input_dim)`
        Returns
            fourier_features (torch.Tensor) : `ff_feature.shape == (B, -1, 2*ff_dim)`
        """

        if self.ff_config.type in ["deterministic_transinr", "deterministic_transinr_nerf"]:
            fourier_features = torch.matmul(coord.unsqueeze(-1), self.ff_linear.unsqueeze(0))
            fourier_features = fourier_features.view(*coord.shape[:-1], -1)
        else:
            fourier_features = torch.matmul(coord, self.ff_linear)

        if not self.ff_config.type == "deterministic_transinr_nerf":
            fourier_features = fourier_features * np.pi

        fourier_features = [torch.cos(fourier_features), torch.sin(fourier_features)]
        fourier_features = torch.cat(fourier_features, dim=-1)
        return fourier_features

    def forward(self, coord, modulation_params_dict=None):
        """Computes the value for each coordination
        Note: `assert outputs.shape[:-1] == coord.shape[:-1]`

        Args
            coord (torch.Tensor): input coordinates to be inferenced
            modulation_params_dict (torch.nn.Parameters): the dictionary of modulation parameters.
                the keys have to be matched with the keys of self.params_dict
                If `modulation_params_dict` given, self.params_dict is modulated before inference.
                If `modulation_params_dict=None`, the inference is conducted based on base params.

        Returns
            outputs (torch.Tensor): evaluated values by INR
        """
        if modulation_params_dict is not None:
            self.check_valid_param_keys(modulation_params_dict)

        batch_size, coord_shape, input_dim = coord.shape[0], coord.shape[1:-1], coord.shape[-1]
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        hidden = self.fourier_mapping(coord) if self.use_ff else coord

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(self.params_dict[param_key], "n m -> b n m", b=batch_size)

            if (modulation_params_dict is not None) and (param_key in modulation_params_dict.keys()):
                modulation_param = modulation_params_dict[param_key]
            else:
                if self.config.use_bias:
                    modulation_param = torch.ones_like(base_param[:, :-1])
                else:
                    modulation_param = torch.ones_like(base_param)

            if self.config.use_bias:
                ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
                hidden = torch.cat([hidden, ones], dim=-1)
                base_param_w, base_param_b = base_param[:, :-1, :], base_param[:, -1:, :]
                if self.ignore_base_param_dict[param_key]:
                    base_param_w = 1.0
                param_w = base_param_w * modulation_param
                if self.normalize_weight:
                    param_w = F.normalize(param_w, dim=1)
                modulated_param = torch.cat([param_w, base_param_b], dim=1)
            else:
                if self.ignore_base_param_dict[param_key]:
                    base_param = 1.0
                if self.normalize_weight:
                    modulated_param = F.normalize(base_param * modulation_param, dim=1)
                else:
                    modulated_param = base_param * modulation_param

            hidden = torch.bmm(hidden, modulated_param)

            if idx < (self.config.n_layer - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        outputs = outputs.view(batch_size, *coord_shape, -1)
        return outputs

    @torch.no_grad()
    def compute_modulated_params_dict(self, modulation_params_dict):
        """Computes the modulated parameters from the modulation parameters.

        Args:
            modulation_params_dict (dict[str, torch.Tensor]): The dictionary of modulation parameters.

        Returns:
            modulated_params_dict (dict[str, torch.Tensor]): The dictionary of modulated parameters.
                Contains keys identical to the keys of `self.params_dict` and corresponding per-instance params.
        """
        self.check_valid_param_keys(modulation_params_dict)

        batch_size = list(modulation_params_dict.values())[0].shape[0]

        modulated_params_dict = {}

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(self.params_dict[param_key], "n m -> b n m", b=batch_size)

            if (modulation_params_dict is not None) and (param_key in modulation_params_dict.keys()):
                modulation_param = modulation_params_dict[param_key]
            else:
                if self.config.use_bias:
                    modulation_param = torch.ones_like(base_param[:, :-1])
                else:
                    modulation_param = torch.ones_like(base_param)

            if self.config.use_bias:
                base_param_w, base_param_b = base_param[:, :-1, :], base_param[:, -1:, :]
                if self.ignore_base_param_dict[param_key]:
                    base_param_w = 1.0
                param_w = base_param_w * modulation_param
                if self.normalize_weight:
                    param_w = F.normalize(param_w, dim=1)
                modulated_param = torch.cat([param_w, base_param_b], dim=1)
            else:
                if self.ignore_base_param_dict[param_key]:
                    base_param = 1.0
                if self.normalize_weight:
                    modulated_param = F.normalize(base_param * modulation_param, dim=1)
                else:
                    modulated_param = base_param * modulation_param

            modulated_params_dict[param_key] = modulated_param

        return modulated_params_dict

    def forward_with_params(self, coord, params_dict):
        """Computes the value for each coordinate, according to INRs with given modulated parameters.

        Args:
            coord (torch.Tensor): Input coordinates in shape (B, ...).
            params_dict (dict[str, torch.Tensor]): The dictionary of modulated parameters.
                Each parameter in `params_dict` must be per-instance (must be in shape (B, fan_in, fan_out)).

        Returns:
            outputs (torch.Tensor): Evaluated values by INRs with per-instance params `params_dict`.
        """
        self.check_valid_param_keys(params_dict)

        batch_size, coord_shape, input_dim = coord.shape[0], coord.shape[1:-1], coord.shape[-1]
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        hidden = self.fourier_mapping(coord) if self.use_ff else coord

        for idx in range(self.config.n_layer):
            param_key = f"linear_wb{idx}"

            modulated_param = params_dict[param_key]
            assert batch_size == modulated_param.shape[0]  # params_dict must contain per-sample params!!

            if self.config.use_bias:
                ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
                hidden = torch.cat([hidden, ones], dim=-1)

            hidden = torch.bmm(hidden, modulated_param)

            if idx < (self.config.n_layer - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        outputs = outputs.view(batch_size, *coord_shape, -1)

        return outputs
