import torch
import torch.nn as nn
import numpy as np

from .modules.coord_sampler import CoordSampler
from .configs import LowRankModulatedTransINRNerfConfig

from .modules.data_encoder import DataEncoder
from .modules.hyponet import HypoNet
from .modules.latent_mapping import LatentMapping
from .modules.weight_groups import WeightGroups
from ..layers import AttentionStack


def volumn_rendering(raw_outputs, z_values):
    """
    This helper function returns rgb mapping based on the four-channeled outpus of INR networks.
    The rgb values are computed based on the predictions for color and transparency at each position.
    """
    rgb, sigma_a = raw_outputs[..., :3], raw_outputs[..., 3]
    rgb = torch.sigmoid(rgb)  # (b, n, p, 3)
    sigma_a = torch.nn.functional.relu(sigma_a)  # (b, n, p)

    dists = torch.cat([z_values[:, 1:] - z_values[:, :-1], torch.ones_like(z_values[:, -1:]) * 1e-3], dim=-1)  # n p
    alpha = 1.0 - torch.exp(-sigma_a * dists)  # b n p
    trans = torch.clamp(1.0 - alpha + 1e-10, max=1.0)  # b n p
    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], dim=-1)
    weights = alpha * torch.cumprod(trans, dim=-1)  # b n p

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    acc_map = torch.sum(weights, dim=-1)
    rgb_map = rgb_map + (1.0 - acc_map).unsqueeze(-1)  # white background

    return rgb_map


class LowRankModulatedTransINRNerf(nn.Module):
    r"""
    `class LowRankModulatedTransINR` is the transformer to predict the Instance Pattern Composers
    to modulate a hyponetwork with a coordinate-based MLP.
    After the transformer predicts the instance pattern composers, which is one factorized weight matrix,
    one layer of the coordinate-based MLP is modulated, while the remaining weights are shared across data.
    Please refer to https://arxiv.org/abs/2211.13223 for more details.
    """
    Config = LowRankModulatedTransINRNerfConfig

    def __init__(self, config: LowRankModulatedTransINRNerfConfig):
        super().__init__()
        self.config = config = config.copy()  # type: LowRankModulatedTransINRNerfConfig
        self.hyponet_config = config.hyponet

        self.coord_sampler = CoordSampler(config.coord_sampler)

        self.encoder = DataEncoder(config.data_encoder)  # DataEncoder have to be developed
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
        self.group_modulation_postfc = nn.ModuleDict()  # pass nn.Linear(embed_dim, shape[0]-1)
        for name, shape in self.hyponet.params_shape_dict.items():
            if name not in self.weight_groups.group_idx_dict:
                continue
            # if a weight matrix of hyponet is modulated, this model does not use `base_param` in the hyponet.
            self.hyponet.ignore_base_param_dict[name] = True

            postfc_input_dim = self.config.transformer.embed_dim
            postfc_output_dim = (shape[0] - 1) if self.hyponet.use_bias else shape[0]

            rank = self.weight_groups.num_groups_dict[name]
            fan_in = (shape[0] - 1) if self.hyponet.use_bias else shape[0]
            fan_out = shape[1]
            shared_factor = torch.randn(1, rank, fan_out) / np.sqrt(rank * fan_in)
            self.shared_factor[name] = nn.Parameter(shared_factor)
            self.group_modulation_postfc[name] = nn.Sequential(
                nn.LayerNorm(postfc_input_dim), nn.Linear(postfc_input_dim, postfc_output_dim)
            )

    def forward(
        self,
        support_imgs,
        support_poses,
        support_focals,
        z_values=None,
        z_range=(2, 6),
        query_rays_o=None,
        query_rays_d=None,
        coord=None,
        return_rgb=True,
        is_train=False,
    ):
        if query_rays_o is None and coord is None:
            raise ValueError("one of query_rays and coord has to be given.")

        batch_size = support_imgs.shape[0]

        xs_emb = self.encode(support_imgs, support_poses, support_focals)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        # predict all pixels of coord after applying the modulation_parms into hyponet
        if coord is None:
            coord, z_values = self.sample_coord_input(
                query_rays_o, query_rays_d, z_range, augment=is_train, with_z_values=True
            )
        raw_outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)
        outputs = raw_outputs
        if return_rgb:
            outputs = volumn_rendering(outputs, z_values)

        return outputs

    def predict_group_modulations(self, group_output):

        modulation_params_dict = dict()

        for name in self.hyponet.params_dict.keys():
            if name not in self.weight_groups.group_idx_dict:
                continue
            start_idx, end_idx = self.weight_groups.group_idx_dict[name]
            _group_output = group_output[:, start_idx:end_idx]

            shape = self.hyponet.params_shape_dict[name]
            fan_out = shape[1]

            # post fc convert the transformer outputs into modulation weights
            _modulation = self.group_modulation_postfc[name](_group_output)

            _modulation_in = _modulation.transpose(-1, -2)  # (B, fan_in, group_size)
            _modulation_out = self.shared_factor[name]  # (1, group_size, fan_out)
            _modulation_out = _modulation_out.repeat(_modulation_in.shape[0], 1, 1)  # (B, group_size, fan_out)

            _modulation = torch.bmm(_modulation_in, _modulation_out)  # (B, fan_in, fan_out)
            modulation_params_dict[name] = _modulation
        return modulation_params_dict

    def encode(self, support_imgs, support_poses, support_focals, put_channels_last=True):
        return self.encoder(support_imgs, support_poses, support_focals, put_channels_last)

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

    def sample_coord_input(
        self, rays_o, rays_d, z_range=(2, 6), num_points_per_ray=None, augment=False, device=None, with_z_values=False
    ):
        device = device if device is not None else rays_o.device
        coord_inputs = self.coord_sampler(rays_o, rays_d, z_range, num_points_per_ray, augment, device, with_z_values)
        return coord_inputs

    def forward_by_subbatch_ray(
        self,
        support_imgs,
        support_poses,
        support_focals,
        z_range=(2, 6),
        query_rays_o=None,
        query_rays_d=None,
        return_rgb=True,
        ray_subbatch_size=1024,
        is_train=False,
    ):
        """
        This module equivalent to forward, but divides the rays into subbatches for managing memory usage.
        `forward_by_subbatch_ray` divides the rays into `ray_subbatch_size` number of subsets.
        Then, iterative forward the model by each subset of rays, and concats all results.
        """

        if query_rays_o is None and coord is None:
            raise ValueError("one of query_rays and coord has to be given.")

        batch_size = support_imgs.shape[0]

        xs_emb = self.encode(support_imgs, support_poses, support_focals)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        outputs_list = []

        for idx in range(0, query_rays_o.shape[1], ray_subbatch_size):
            subrays_o = query_rays_o[:, idx : idx + ray_subbatch_size]
            subrays_d = query_rays_d[:, idx : idx + ray_subbatch_size]

            coord, z_values = self.sample_coord_input(
                subrays_o, subrays_d, z_range, augment=is_train, with_z_values=True
            )

            raw_outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)
            outputs = raw_outputs
            if return_rgb:
                outputs = volumn_rendering(outputs, z_values)

            outputs_list.append(outputs)

        outputs_total = torch.cat(outputs_list, dim=1)
        return outputs_total

    def predict_modulation_params_dict(self, support_imgs, support_poses, support_focals):
        """Computes the modulation parameters for given inputs."""
        batch_size = support_imgs.shape[0]

        xs_emb = self.encode(support_imgs, support_poses, support_focals)
        xs_latent = self.encode_latent(xs_emb)  # latent mapping
        weight_token_input = self.weight_groups(batch_size=batch_size)  # (B, num_groups_total, embed_dim)

        transformer_input = torch.cat([xs_latent, weight_token_input], dim=1)
        transformer_output = self.transformer(transformer_input)

        transformer_output_groups = transformer_output[:, -self.num_group_total :]

        # returns the weights for modulation of hypo-network
        modulation_params_dict = self.predict_group_modulations(transformer_output_groups)

        return modulation_params_dict

    def predict_hyponet_params_dict(self, support_imgs, support_poses, support_focals):
        """Computes the modulated parameters of hyponet for given inputs."""
        modulation_params_dict = self.predict_modulation_params_dict(support_imgs, support_poses, support_focals)
        params_dict = self.hyponet.compute_modulated_params_dict(modulation_params_dict)
        return params_dict

    def forward_with_params(
        self,
        query_rays_o,
        query_rays_d,
        z_range=((2, 6),),
        return_rgb=True,
        ray_subbatch_size=1024,
        is_train=False,
        modulation_params_dict=None,
        hyponet_params_dict=None,
    ):
        """Computes the outputs according to INRs specified with either modulation parameters or modulated parameters.
        Note: Exactly one of `modulation_params_dict` or `hyponet_params_dict` must be given.

        Args:
            query_rays_o (torch.Tensor): Origins of query rays.
            query_rays_d (torch.Tensor): Directions of query rays.
            z_range (Tuple[int, int]): Range of z values.
            return_rgb (bool): Whether to return in rgb. If True, returns the rendered result.
            ray_subbatch_size (int): Size of a subbatch.
            is_train (bool): If True, z values are augmented.
            modulation_params_dict (dict[str, torch.Tensor], optional): Modulation parameters.
            hyponet_params_dict (dict[str, torch.Tensor], optional): Modulated hyponet parameters.
        Returns:
            outputs (torch.Tensor): Computed outputs according to INRs with specified modulation/modulated params.
        """
        if (modulation_params_dict is None) and (hyponet_params_dict is None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")
        if (modulation_params_dict is not None) and (hyponet_params_dict is not None):
            raise ValueError("Exactly one of modulation_params_dict or hyponet_params_dict must be given")

        outputs_list = []

        for idx in range(0, query_rays_o.shape[1], ray_subbatch_size):
            subrays_o = query_rays_o[:, idx : idx + ray_subbatch_size]
            subrays_d = query_rays_d[:, idx : idx + ray_subbatch_size]

            coord, z_values = self.sample_coord_input(
                subrays_o, subrays_d, z_range, augment=is_train, with_z_values=True
            )

            if modulation_params_dict is None:
                assert hyponet_params_dict is not None
                raw_outputs = self.hyponet.forward_with_params(coord, params_dict=hyponet_params_dict)
            else:
                assert hyponet_params_dict is None
                raw_outputs = self.hyponet.forward(coord, modulation_params_dict=modulation_params_dict)

            outputs = raw_outputs
            if return_rgb:
                outputs = volumn_rendering(outputs, z_values)

            outputs_list.append(outputs)

        outputs_total = torch.cat(outputs_list, dim=1)
        return outputs_total
