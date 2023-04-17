import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from .utils import convert_int_to_list


class Unfold1D(nn.Module):
    """unfold audio (3D tensor), where the channel dimension is one."""

    def __init__(self, patch_size=200, use_padding=True, padding_type="reflect"):
        super().__init__()
        self.patch_size = patch_size
        self.use_padding = use_padding
        self.padding_type = padding_type

    def forward(self, data):
        assert data.ndim == 3  # audio
        if self.use_padding:
            if not data.shape[-1] % self.patch_size == 0:
                len_data = data.shape[-1]
                pad_size = self.patch_size - (len_data % self.patch_size)
            else:
                pad_size = 0
        else:
            pad_size = 0
        data = torch.nn.functional.pad(data, (0, pad_size), mode=self.padding_type)

        num_patches = data.shape[-1] // self.patch_size
        data = torch.reshape(data, (data.shape[0], num_patches, -1))
        return data


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
        elif self.type == "unfold_audio":
            self.encoder = Unfold1D(spec.patch_size, spec.use_padding)
            self.output_dim = self.input_dim * self.encoder.patch_size
            self.is_encoder_out_channels_last = True
        else:
            # If necessary, implement additional wrapper for extracting features of data
            raise NotImplementedError

        if not self.trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, xs, put_channels_last=False):
        xs_embed = self.encoder(xs)
        if put_channels_last and not self.is_encoder_out_channels_last:
            permute_idx_range = [i for i in range(2, xs_embed.ndim)]
            return xs_embed.permute(0, *permute_idx_range, 1).contiguous()
        else:
            return xs_embed
