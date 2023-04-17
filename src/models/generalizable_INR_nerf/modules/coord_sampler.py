import einops
import torch
import torch.nn as nn


class CoordSampler(nn.Module):
    """Generates coordinate inputs according to the given data type.
    This class can be more implemented according to the coordinates sampling strategy.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.coord_range = config.coord_range
        self.num_points_per_ray = config.num_points_per_ray

    def base_sampler(self, rays_o, rays_d, z_range, augment, num_points_per_ray=None, with_z_values=False, device=None):
        device = device if device is not None else rays_o.device
        z_near, z_far = z_range[0, 0], z_range[0, 1]

        batch_size = rays_o.shape[0]
        rays_o_dim, rays_d_dim = rays_o.shape[-1], rays_d.shape[-1]
        assert rays_o_dim == rays_d_dim  # 3

        rays_o = rays_o.view(batch_size, -1, 3)
        rays_d = rays_d.view(batch_size, -1, 3)

        num_rays = rays_o.shape[1]

        z_values = torch.linspace(z_near, z_far, num_points_per_ray, device=device)
        z_values = einops.repeat(z_values, "p -> n p", n=num_rays)

        # if augment, perturb the z_values within each z interval
        if augment:
            delta_ray = (z_far - z_near) / (num_points_per_ray - 1)
            z_values = z_values + torch.rand(num_rays, num_points_per_ray, device=device) * delta_ray

        rays_o = rays_o.view(batch_size, num_rays, 1, rays_o_dim)
        rays_d = rays_d.view(batch_size, num_rays, 1, rays_d_dim)
        z_values_coord = z_values.view(1, num_rays, num_points_per_ray, 1)

        coords = rays_o + rays_d * z_values_coord
        if with_z_values:
            return coords, z_values
        else:
            return coords, None

    def forward(
        self, rays_o, rays_d, z_range, num_points_per_ray=None, augment=False, device=None, with_z_values=False
    ):
        if num_points_per_ray is None:
            num_points_per_ray = self.num_points_per_ray

        coords, z_values = self.base_sampler(
            rays_o, rays_d, z_range, augment, num_points_per_ray, with_z_values, device
        )

        if with_z_values:
            return coords, z_values
        else:
            return coords
