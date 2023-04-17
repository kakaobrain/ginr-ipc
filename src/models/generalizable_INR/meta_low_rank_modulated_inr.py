import typing

import einops
import numpy as np
import torch
import torch.nn as nn

from .configs import MetaLowRankModulatedINRConfig
from .modules.coord_sampler import CoordSampler
from .modules.hyponet import HypoNet
from .transinr import TransINR

Tensor = torch.Tensor
TensorDict = typing.Dict[str, Tensor]


def repeat_along_batch_dim(tensor: Tensor, batch_size: int):
    return einops.repeat(tensor, "... -> b ...", b=batch_size)


class ModulatedParamsFactors(nn.Module):
    def __init__(
        self,
        params_shape_dict,
        use_bias_in_hyponet,
        ranks,
        modulated_layer_idxs=None,
        use_factorization=True,
    ):
        r"""Class to decompose each modulation parameter W into matrix multiplication of shared factor U and
        instance-specific modulation factor V.

        V is adapted via gradient descent during inner loop, while U and init value of V are trained in outer loop.

        Arguments:
            params_shape_dict: Dictionary of hyponet parameter shapes.
            use_bias_in_hyponet: Whether the hyponet uses bias.
            ranks (list of int): Ranks of each factorization (i.e. fan_in of U and fan_out of V).
                Irrelevant if `use_factorization` is True.
            modulated_layer_idxs: List of modulated layer indices.
            use_factorization: If True, W is factorized into U and V. If False, W = V and shared factor U is set `None`.
        """
        super().__init__()

        if len(ranks) == 1:
            ranks = [ranks[0] for _ in range(len(params_shape_dict))]
        else:
            assert len(ranks) == len(params_shape_dict)

        if modulated_layer_idxs is None:
            modulated_layer_idxs = list(range(len(params_shape_dict)))
        else:
            assert len(modulated_layer_idxs) > 0

        self.init_modulation_factors = nn.ParameterDict()
        self.shared_factors = nn.ParameterDict()

        for idx, (name, shape) in enumerate(params_shape_dict.items()):
            if idx not in modulated_layer_idxs:
                continue
            fan_in = (shape[0] - 1) if use_bias_in_hyponet else shape[0]
            fan_out = shape[1]
            rank = min(ranks[idx], fan_out)

            if use_factorization:
                init_modulation_factor = torch.randn(fan_in, rank)
                init_shared_factor = torch.randn(rank, fan_out) / np.sqrt(rank * fan_in)

                self.init_modulation_factors[name] = nn.Parameter(init_modulation_factor)
                self.shared_factors[name] = nn.Parameter(init_shared_factor)
            else:
                # rank is irrelevant in this case
                init_modulation_factor = torch.randn(fan_in, fan_out) / np.sqrt(fan_in)

                self.init_modulation_factors[name] = nn.Parameter(init_modulation_factor)
                self.shared_factors[name] = None

    def compute_modulation_params_dict(self, modulation_factors_dict):
        r"""Computes modulation param W by multiplying shared factor U and modulation factor V.
        If shared factor U is None (i.e. `use_factorization` was False), W = V.
        """
        modulation_params_dict = {}

        for name, modulation_factor in modulation_factors_dict.items():
            shared_factor = self.shared_factors[name]
            if shared_factor is not None:
                shared_factor = repeat_along_batch_dim(shared_factor, batch_size=modulation_factor.shape[0])
                modulation_param = torch.bmm(modulation_factor, shared_factor)
            else:
                modulation_param = modulation_factor
            modulation_params_dict[name] = modulation_param

        return modulation_params_dict

    @property
    def modulated_param_names(self):
        return list(self.shared_factors.keys())


class MetaLowRankModulatedINR(TransINR):
    r"""
    `class MetaLowRankModulatedINR` is an optimization-based meta-learner for INR modulation.
    While only one weight matrix is adapted to each data instance for modulating a hyponetwork with a coordinate-based MLP, 
    the remaining weights are trained over data during outer loop.
    Please refer to Algorithm 1 in our paper (https://arxiv.org/abs/2211.13223) for more details.
    """
    Config = MetaLowRankModulatedINRConfig

    def __init__(self, config: MetaLowRankModulatedINRConfig):
        super(TransINR, self).__init__()

        self.config = config

        self.coord_sampler = CoordSampler(config.coord_sampler)
        self.hyponet = HypoNet(config.hyponet)

        self.factors = ModulatedParamsFactors(
            self.hyponet.params_shape_dict,
            use_bias_in_hyponet=self.hyponet.use_bias,
            ranks=config.rank,
            modulated_layer_idxs=config.modulated_layer_idxs,
            use_factorization=config.use_factorization,
        )

        for name in self.factors.modulated_param_names:
            # We always ignore base param so that each modulated weight W is directly computed
            self.hyponet.ignore_base_param_dict[name] = True

        self.n_inner_step = self.config.n_inner_step
        self.inner_lr = self.config.inner_lr

    @torch.enable_grad()
    def get_init_modulation_factors(self, xs: Tensor):
        r"""Returns the initial modulation factors."""
        modulation_factors_dict = self.factors.init_modulation_factors
        modulation_factors_dict = {
            name: repeat_along_batch_dim(factor, xs.shape[0]) for name, factor in modulation_factors_dict.items()
        }
        return modulation_factors_dict

    def predict_with_modulation_factors(self, xs, modulation_factors_dict, coord=None):
        r"""Inference function on Hyponet, modulated via given modulation factors."""
        coord = self.sample_coord_input(xs) if coord is None else coord

        # convert modulation factors into modulation params
        modulation_params_dict = self.factors.compute_modulation_params_dict(modulation_factors_dict)

        # predict all pixels of coord after applying the modulation_parms into hyponet
        outputs = self.hyponet(coord, modulation_params_dict=modulation_params_dict)
        permute_idx_range = [i for i in range(1, xs.ndim - 1)]
        outputs = outputs.permute(0, -1, *permute_idx_range)
        return outputs

    def inner_step(
        self,
        xs: Tensor,
        modulation_factors_dict: TensorDict,
        coord: Tensor,
        inner_lr: Tensor,
        is_training: bool = True,
    ):
        r"""Single adaptation step of modulation factors via SGD w.r.t. the reconstruction loss for `xs`."""

        with torch.enable_grad():
            # compute reconstruction
            recons = self.predict_with_modulation_factors(xs, modulation_factors_dict, coord)

            # compute the loss
            # reduction should be "sum" here, since we are computing per-sample gradient
            metrics = self.compute_loss(recons, xs, reduction="sum")

            # compute gradient w.r.t. latents
            factor_names = list(modulation_factors_dict.keys())
            modulation_factors_list = list(modulation_factors_dict.values())
            grads_list = torch.autograd.grad(metrics["loss_total"], modulation_factors_list, create_graph=is_training)

            # take an SGD step
            new_modulation_factors_dict = {}
            for name, factor, grad in zip(factor_names, modulation_factors_list, grads_list):
                if self.config.hyponet.normalize_weight:
                    lr_scale = factor.norm(dim=[1, 2], keepdim=True).pow(2.0)
                else:
                    lr_scale = 1.0
                new_factor = factor - inner_lr * lr_scale * grad
                new_modulation_factors_dict[name] = new_factor

        # only for logging
        logs = {
            **{f"{key}_mod": value.detach().clone() for key, value in modulation_factors_dict.items()},
            "recons": recons.detach().clone(),
            "loss_total": metrics["loss_total"].detach().clone(),
            "mse": metrics["mse"].detach().clone(),
            "psnr": metrics["psnr"].detach().clone(),
        }
        return new_modulation_factors_dict, logs

    def inner_loop(self, xs, n_inner_step=1, inner_lr=0.1, is_training=True):
        r"""A loop of latent adaptation steps, served as the inner loop of meta-learning."""

        # We assume that inner loop uses the coords of shape identical to the spatial shape of xs, while not using
        # coordinate subsampling. For this reason, we compute `coord` from `xs` in the inner loop.
        coord = self.sample_coord_input(xs)
        modulation_factors_dict = self.get_init_modulation_factors(xs)

        inner_loop_history = []

        for step_idx in range(n_inner_step):
            modulation_factors_dict, logs = self.inner_step(
                xs, modulation_factors_dict, coord, inner_lr, is_training=is_training
            )
            inner_loop_history.append(logs)

        return modulation_factors_dict, inner_loop_history

    def _collate_inner_loop_history(self, inner_loop_history):
        r"""Reorganize `inner_loop_history` which is list of dicts for logging from each inner step.
        Metrics (scalars) are stacked along dim=0, while other tensors (images or modulation factors) are
        stacked along dim=1. Returns the dictionary which looks like:
            {
                ...
                "recons": tensor of shape (batch_size, n_inner_step, 3, H, W)
                "psnr": tensor of shape (n_inner_step,)
                ...
            }
        """
        keys = inner_loop_history[0].keys()
        collated = {}
        for key in keys:
            tensors = [dict_[key] for dict_ in inner_loop_history]
            is_scalar = tensors[0].ndim == 0
            if is_scalar:
                tensors = torch.stack(tensors, dim=0)
            else:
                tensors = torch.stack(tensors, dim=1)
            collated[key] = tensors
        return collated

    def forward(self, xs, coord=None, n_inner_step=None, inner_lr=None, is_training=None):
        r"""Infers the signal values at the given coordinates, after an inner loop adapted for `xs`.

        Arguments:
            xs (Tensor): data which is used to adapt latents.
            coord (Tensor, optional): coordinates to infer signal values.
            n_inner_step (int, optional): number of inner steps. (Default: `self.n_inner_step`)
            inner_lr (float, optional): learning rate used in inner steps. (Default: `self.inner_lr`)
            is_training (bool, optional): indicates whether it is in training context. (Default: `self.training`)
        """
        coord = self.sample_coord_input(xs) if coord is None else coord
        n_inner_step = self.n_inner_step if n_inner_step is None else n_inner_step
        inner_lr = self.inner_lr if inner_lr is None else inner_lr
        is_training = self.training if is_training is None else is_training

        modulation_factors_dict, inner_loop_history = self.inner_loop(
            xs, n_inner_step=n_inner_step, inner_lr=inner_lr, is_training=is_training
        )
        outputs = self.predict_with_modulation_factors(xs, modulation_factors_dict, coord)
        collated_history = self._collate_inner_loop_history(inner_loop_history)
        return outputs, modulation_factors_dict, collated_history
