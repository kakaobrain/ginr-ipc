# reference: https://github.com/yinboc/trans-inr/blob/master/utils/geometry.py

import torch


def poses_to_rays(poses, focal, image_h, image_w):
    """
    Pose columns are: 3 camera axes specified in world coordinate + 1 camera position.
    Camera: x-axis right, y-axis up, z-axis inward.
    Focal is in pixel-scale.
    Args:
        poses: (... 3 4)
        focal: (... 2)
    Returns:
        rays_o, rays_d: shape (... image_h image_w 3)
    """
    device = poses.device
    bshape = poses.shape[:-2]
    poses = poses.view(-1, 3, 4)
    focal = focal.view(-1, 2)
    bsize = poses.shape[0]

    x, y = torch.meshgrid(
        torch.arange(image_w, device=device), torch.arange(image_h, device=device), indexing="xy"
    )  # h w
    x, y = x + 0.5, y + 0.5  # modified to + 0.5
    x, y = x.unsqueeze(0), y.unsqueeze(0)  # h w -> 1 h w
    focal = focal.unsqueeze(1).unsqueeze(1)  # b 2 -> b 1 1 2
    dirs = torch.stack(
        [
            (x - image_w / 2) / focal[..., 0],
            -(y - image_h / 2) / focal[..., 1],
            -torch.ones(bsize, image_h, image_w, device=device),
        ],
        dim=-1,
    )  # b h w 3

    poses = poses.unsqueeze(1).unsqueeze(1)  # b 3 4 -> b 1 1 3 4
    rays_o = poses[..., -1].repeat(1, image_h, image_w, 1)  # b h w 3
    rays_d = (dirs.unsqueeze(-2) * poses[..., :3]).sum(dim=-1)  # b h w 3

    rays_o = rays_o.view(*bshape, *rays_o.shape[1:])
    rays_d = rays_d.view(*bshape, *rays_d.shape[1:])

    return rays_o, rays_d
