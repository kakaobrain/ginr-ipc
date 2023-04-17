import einops
import numpy as np
import torch
import torch.nn as nn
from utils.geometry import poses_to_rays
from .utils import convert_int_to_list


class Unfold(nn.Module):
    """Note: only 4D tensors are currently supported by pytorch."""

    def __init__(self, patch_size, padding=0, img_channel=3):
        super().__init__()
        self.patch_size = convert_int_to_list(patch_size, len_list=2)
        self.padding = convert_int_to_list(padding, len_list=2)

        self.unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=self.padding)

    def forward(self, data):
        """
        Args
            data (torch.tensor): data with shape = [batch_size, channel, ...]
        Returns
            unfolded_data (torch.tensor): unfolded data
                shape = [batch_size, channel * patch_size[0] * patch_size[1], L]
        """
        unfolded_data = self.unfold(data)
        return unfolded_data


class DataEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = config.type
        self.trainable = config.trainable
        self.input_dim = config.n_channel
        self.output_dim = None

        spec = config.encoder_spec
        if self.type == "unfold":
            self.encoder = Unfold(spec.patch_size, spec.padding)
            self.output_dim = self.input_dim * np.product(self.encoder.patch_size)
            self.is_encoder_out_channels_last = False
        else:
            # If necessary, implement additional wrapper for extracting features of data
            raise NotImplementedError

        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, support_imgs, support_poses, support_focals, put_channels_last=False):
        """
        the start point and normalized direction are concatenated into the color channel or images,
        and then the output images have nine-channels.
        """
        batch_size = support_imgs.shape[0]
        height, width = support_imgs.shape[-2:]

        rays_o, rays_d = poses_to_rays(support_poses, support_focals, height, width)
        rays_o = einops.rearrange(rays_o, "b n h w c -> b n c h w")
        rays_d = einops.rearrange(rays_d, "b n h w c -> b n c h w")

        xs = torch.cat([support_imgs, rays_o, rays_d], dim=2)  # channel-wise concatenation
        xs = einops.rearrange(xs, "b n d h w -> (b n) d h w")
        xs_embed = self.encoder(xs)
        xs_embed = einops.rearrange(xs_embed, "(b n) ppd l -> b (n l) ppd", b=batch_size)

        if put_channels_last and not self.is_encoder_out_channels_last:
            # here, we have used einops.rearrange to consider channel_last type by default
            return xs_embed
        else:
            permute_idx_range = [i for i in range(2, xs_embed.ndim)]
            return xs_embed.permute(0, *permute_idx_range, 1).contiguous()
