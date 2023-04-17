import torch
import torch.nn as nn


def shape2coordinate(spatial_shape, batch_size, min_value=-1.0, max_value=1.0, upsample_ratio=1, device=None):
    coords = []
    for num_s in spatial_shape:
        num_s = int(num_s * upsample_ratio)
        _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
        _coords = min_value + (max_value - min_value) * _coords
        coords.append(_coords)
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords, dim=-1)
    ones_like_shape = (1,) * coords.ndim
    coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    return coords


class CoordSampler(nn.Module):
    """Generates coordinate inputs according to the given data type.
    This class can be more implemented according to the coordinates sampling strategy.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_type = config.data_type
        assert self.data_type in ["image", "audio"]

        self.coord_range = config.coord_range

    def base_sampler(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        coord_range = self.coord_range if coord_range is None else coord_range
        min_value, max_value = coord_range

        if self.data_type in ["image", "audio"]:
            if self.data_type == "image":
                if xs.ndim == 3:
                    xs = xs.unsqueeze(0)
                else:
                    assert xs.ndim == 4
                batch_size, spatial_shape = xs.shape[0], xs.shape[-2:]

            elif self.data_type == "audio":
                if xs.ndim == 2:
                    xs = xs.unsqueeze(0)
                else:
                    assert xs.ndim == 3
                batch_size, spatial_shape = xs.shape[0], xs.shape[-1:]

            return shape2coordinate(spatial_shape, batch_size, min_value, max_value, upsample_ratio, device)
        else:
            raise NotImplementedError

    def forward(self, xs, coord_range=None, upsample_ratio=1.0, device=None):
        coords = self.base_sampler(xs, coord_range, upsample_ratio, device)
        return coords
