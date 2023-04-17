import numpy as np
import torch
import torch.nn as nn

from .configs import LowRankModulatedTransINRConfig
from .modules.coord_sampler import CoordSampler
from .modules.data_encoder import DataEncoder
from .modules.hyponet import HypoNet
from .modules.latent_mapping import LatentMapping
from .modules.weight_groups import WeightGroups
from ..layers import AttentionStack


class LowRankModulatedTransINR(nn.Module):
    r"""
    `class LowRankModulatedTransINR` is the transformer to predict the Instance Pattern Composers
    to modulate a hyponetwork with a coordinate-based MLP.
    After the transformer predicts the instance pattern composers, which is one factorized weight matrix,
    one layer of the coordinate-based MLP is modulated, while the remaining weights are shared across data.
    Please refer to https://arxiv.org/abs/2211.13223 for more details.
    """
    Config = LowRankModulatedTransINRConfig

    def __init__(self, config: LowRankModulatedTransINRConfig):
        super().__init__()
        self.config = config = config.copy()
        self.hyponet_config = config.hyponet

        self.coord_sampler = CoordSampler(config.coord_sampler)
        self.encoder = DataEncoder(config.data_encoder)
        self.latent_mapping = LatentMapping(config.latent_mapping, input_dim=self.encoder.output_dim)
        self.transformer = AttentionStack(config.transformer)
        self.hyponet = HypoNet(config.hyponet)

        self.weight_groups = WeightGroups(
            self.hyponet.params_shape_dict,
            num_groups=config.n_weight_groups,
            weight_dim=config.transformer.embed_dim,
            modulated_layer_idxs=config.modulated_layer_idxs,
        )

        self.num_group_total = self.weight_groups.num_group_total
        self.shared_factor = nn.ParameterDict()
        self.group_modulation_postfc = nn.ModuleDict()
        for name, shape in self.hyponet.params_shape_dict.items():
            if name not in self.weight_groups.group_idx_dict:
                continue
            # if a weight matrix of hyponet is modulated, this model does not use `base_param` in the hyponet.
            self.hyponet.ignore_base_param_dict[name] = True

            rank = self.weight_groups.num_groups_dict[name]
            fan_in = (shape[0] - 1) if self.hyponet.use_bias else shape[0]
            fan_out = shape[1]

            shared_factor = torch.randn(1, rank, fan_out) / np.sqrt(rank * fan_in)
            self.shared_factor[name] = nn.Parameter(shared_factor)

            postfc_input_dim = self.config.transformer.embed_dim
            postfc_output_dim = (shape[0] - 1) if self.hyponet.use_bias else shape[0]
            self.group_modulation_postfc[name] = nn.Sequential(
                nn.LayerNorm(postfc_input_dim), nn.Linear(postfc_input_dim, postfc_output_dim)
            )

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif module.bias is not None:
            module.bias.data.zero_()

    def forward(self, xs, coord=None, keep_xs_shape=True):
        r"""
        Args:
            xs (torch.Tensor): (B, input_dim, *xs_spatial_shape)
            coord (torch.Tensor): (B, *coord_spatial_shape)
            keep_xs_shape (bool): If True, the outputs of hyponet (MLPs) is permuted and reshaped as `xs`
              If False, it returns the outputs of MLPs with channel_last data type (i.e. `outputs.shape == coord.shape`)
        Returns:
            outputs (torch.Tensor): `assert outputs.shape == xs.shape`
        """
        batch_size = xs.shape[0]
        coord = self.sample_coord_input(xs) if coord is None else coord
        xs_emb = self.encode(xs)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        # predict all pixels of coord after applying the modulation_parms into hyponet
        outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)
        if keep_xs_shape:
            permute_idx_range = [i for i in range(1, xs.ndim - 1)]
            outputs = outputs.permute(0, -1, *permute_idx_range)
        return outputs

    def predict_group_modulations(self, group_output):
        modulation_params_dict = dict()
        for name in self.hyponet.params_dict.keys():
            if name not in self.weight_groups.group_idx_dict:
                continue
            start_idx, end_idx = self.weight_groups.group_idx_dict[name]
            _group_output = group_output[:, start_idx:end_idx]

            # post fc convert the transformer outputs into modulation weights
            _modulation = self.group_modulation_postfc[name](_group_output)  # (B, group_size, fan_in+fan_out)

            _modulation_in = _modulation.transpose(-1, -2)  # (B, fan_in, group_size)
            _modulation_out = self.shared_factor[name]  # (1, group_size, fan_out)
            _modulation_out = _modulation_out.repeat(_modulation_in.shape[0], 1, 1)  # (B, group_size, fan_out)

            # factorized matrix multiplication for weight modulation
            _modulation = torch.bmm(_modulation_in, _modulation_out)  # (B, fan_in, fan_out)
            modulation_params_dict[name] = _modulation
        return modulation_params_dict

    def encode(self, xs, put_channels_last=True):
        return self.encoder(xs, put_channels_last=put_channels_last)

    def encode_latent(self, xs_embed):
        return self.latent_mapping(xs_embed)

    def compute_loss(self, preds, targets, reduction="mean"):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = torch.reshape((preds - targets) ** 2, (batch_size, -1)).mean(dim=-1)

        if reduction == "mean":
            total_loss = sample_mses.mean()
            psnr = (-10 * torch.log10(sample_mses)).mean()
        elif reduction == "sum":
            total_loss = sample_mses.sum()
            psnr = (-10 * torch.log10(sample_mses)).sum()
        else:
            total_loss = sample_mses
            psnr = -10 * torch.log10(sample_mses)

        return {"loss_total": total_loss, "mse": total_loss, "psnr": psnr}

    def sample_coord_input(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        device = device if device is not None else xs.device
        coord_inputs = self.coord_sampler(xs, coord_range, upsample_ratio, device)
        return coord_inputs

    def predict_modulation_params_dict(self, xs):
        r"""Computes the modulation parameters for given inputs."""
        batch_size = xs.shape[0]
        xs_emb = self.encode(xs)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        return modulation_params_dict

    def predict_hyponet_params_dict(self, xs):
        r"""Computes the modulated parameters of hyponet for given inputs."""
        modulation_params_dict = self.predict_modulation_params_dict(xs)
        params_dict = self.hyponet.compute_modulated_params_dict(modulation_params_dict)
        return params_dict

    def forward_with_params(
        self,
        coord,
        keep_xs_shape=True,
        modulation_params_dict=None,
        hyponet_params_dict=None,
    ):
        r"""Computes the output values for coordinates according to INRs specified with either modulation parameters or
        modulated parameters.
        Note: Exactly one of `modulation_params_dict` or `hyponet_params_dict` must be given.

        Args:
            coord (torch.Tensor): Input coordinates in shape (B, ...)
            keep_xs_shape (bool): If True, the outputs of hyponet (MLPs) is permuted and reshaped as `xs`
              If False, it returns the outputs of MLPs with channel_last data type (i.e. `outputs.shape == coord.shape`)
            modulation_params_dict (dict[str, torch.Tensor], optional): Modulation parameters.
            hyponet_params_dict (dict[str, torch.Tensor], optional): Modulated hyponet parameters.
        Returns:
            outputs (torch.Tensor): Evaluated values according to INRs with specified modulation/modulated parameters.
        """
        if (modulation_params_dict is None) and (hyponet_params_dict is None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")
        if (modulation_params_dict is not None) and (hyponet_params_dict is not None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")

        if modulation_params_dict is None:
            assert hyponet_params_dict is not None
            outputs = self.hyponet.forward_with_params(coord, params_dict=hyponet_params_dict)
        else:
            assert hyponet_params_dict is None
            outputs = self.hyponet.forward(coord, modulation_params_dict=modulation_params_dict)

        if keep_xs_shape:
            permute_idx_range = [i for i in range(1, outputs.ndim - 1)]
            outputs = outputs.permute(0, -1, *permute_idx_range)
        return outputs
