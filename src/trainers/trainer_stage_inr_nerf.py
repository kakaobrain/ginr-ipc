import logging

import einops
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from utils.geometry import poses_to_rays
from .trainer_stage_inr import Trainer as INRTrainer

logger = logging.getLogger(__name__)

# utils for slicing batched data
def batch_slice_and_gather(batch, batch_slice_idx):
    assert batch.ndim == 3
    batch_size = batch.shape[0]
    sliced_batch = []
    for idx in range(batch_size):
        sliced_batch.append(batch[idx : idx + 1, batch_slice_idx[idx]])
    sliced_batch = torch.cat(sliced_batch, dim=0)
    return sliced_batch


class RaySubSampler:
    def __init__(self, subsampler_config):
        r"""`RaySubSampler` subsamples the rays for efficient training.
        The configs are determined by the config yaml file.
        After the subsampling, the rays are flattened and returned.
        """
        self.subsample_type = subsampler_config.type
        self.train_num_rays = subsampler_config.train_num_rays
        self.use_adaptive_sample_ray = subsampler_config.use_adaptive_sample_ray
        self.end_epoch_adaptive_sample_ray = subsampler_config.end_epoch_adaptive_sample_ray

    def subsample(self, rays_o, rays_d, xs, epoch=0, is_train=True):
        r"""This module returns the subsampled rays.
        First, `self.subsample_rays_idxs` subsample the index of ray subsamples,
        and `subsample_rays` and `subsample_xs` extract the selected rays and pixels at the subsampled ray position.
        """
        subsample_ray_idx = self.subsample_rays_idx(rays_o, xs, epoch, is_train)
        rays_o, rays_d = self.subsample_rays(rays_o, rays_d, subsample_ray_idx)
        xs = self.subsample_xs(xs, subsample_ray_idx)
        return rays_o, rays_d, xs

    def subsample_rays_idx(self, rays_input, xs=None, epoch=0, is_train=True):
        if not rays_input.ndim == 3:
            raise ValueError(f"rays have to be flattened before subsampling")
        subsample_type = self.subsample_type
        train_num_rays = self.train_num_rays

        if subsample_type is None or train_num_rays is None:
            return None
        elif subsample_type == "random":
            subsample_ray_idx = self.subsample_random_idx(rays_input)
        elif subsample_type == "adaptive_random":
            assert self.end_epoch_adaptive_sample_ray is not None
            if is_train and epoch <= self.end_epoch_adaptive_sample_ray:
                assert xs is not None
                subsample_ray_idx = self.subsample_adaptive_random_ray(rays_input, xs, train_num_rays)
            else:
                # during evaluation, adaptive random ray sampling is not allowed.
                subsample_ray_idx = self.subsample_random_idx(rays_input)
        else:
            raise NotImplemented(f"Not supported subsample_type: {subsample_type}")

        return subsample_ray_idx

    def subsample_random_idx(self, rays_input):
        batch_size, num_rays, _ = rays_input.shape
        num_subsample = self.train_num_rays

        subsample_idx = []
        for _ in range(batch_size):
            rand_idx = torch.randperm(num_rays, device=rays_input.device)
            rand_idx = rand_idx[:num_subsample]
            subsample_idx.append(rand_idx.unsqueeze(0))
        subsample_idx = torch.cat(subsample_idx, dim=0)
        return subsample_idx

    def subsample_adaptive_random_ray(self, rays_input, xs, num_rays):
        r"""
        According to the original paper of TransINR,
        adaptive random ray subsampling is used in the first few epochs,
        for increasing the stability of training.
        """
        assert xs.ndim == 3  # image has to be flattened before subsampling
        batch_size = rays_input.shape[0]
        num_fg_rays = num_rays // 2
        subsample_idx = []

        for idx in range(batch_size):
            # foreground ray sampling
            fgs = (xs[idx].min(dim=-1).values < 1).nonzero().view(-1).contiguous()
            if num_fg_rays <= len(fgs):
                random_fg_idx = torch.randperm(len(fgs), device=fgs.device)
                random_fg_idx = random_fg_idx[:num_fg_rays]
                fgs = fgs[random_fg_idx]  # foreground indexes
            else:
                # if the number of foregrounds is smaller than the number for sampling
                # randomly duplicate the foregrounds
                random_fg_idx = torch.randint(0, len(fgs), size=(num_fg_rays - len(fgs),), device=fgs.device)
                _fgs = fgs[random_fg_idx]
                fgs = torch.cat([fgs, _fgs], dim=0)

            assert len(fgs) == num_fg_rays

            # remaining rays are randomly selected
            random_remainder_idx = torch.randperm(rays_input.shape[1], device=rays_input.device)
            num_remainder_rays = num_rays - num_fg_rays
            remainder = random_remainder_idx[:num_remainder_rays]
            subsample_idx.append(torch.cat([fgs, remainder], dim=0).unsqueeze(0))

        subsample_idx = torch.cat(subsample_idx, dim=0)
        return subsample_idx

    def subsample_rays(self, rays_o, rays_d, subsample_idx):
        if rays_o.ndim > 3 or rays_d.ndim > 3:
            raise ValueError(f"rays have to be flattened, rays_o: {rays_o.shape}, rays_d: {rays_d.shape}")

        sub_rays_o = batch_slice_and_gather(rays_o, subsample_idx)
        sub_rays_d = batch_slice_and_gather(rays_d, subsample_idx)
        return sub_rays_o, sub_rays_d

    def subsample_xs(self, xs, subsample_idxs):
        if xs.ndim > 3:
            raise ValueError(f"rays have to be flattened before subsampling, xs: {xs.shape}")
        sub_xs = batch_slice_and_gather(xs, subsample_idxs)
        return sub_xs


class Trainer(INRTrainer):
    def __init__(self, *args, **kwargs):
        super(INRTrainer, self).__init__(*args, **kwargs)
        self.ray_subsampler = RaySubSampler(self.config.loss.subsample)

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()
        exp_config = self.config.experiment

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xs in pbar:
            model.zero_grad()
            xs = {key: value.to(self.device, non_blocking=True) for key, value in xs.items()}

            support_imgs, support_poses, support_focals = xs["support_imgs"], xs["support_poses"], xs["support_focals"]
            query_imgs, query_poses, query_focals = xs["query_imgs"], xs["query_poses"], xs["query_focals"]
            z_range = xs["z_range"]

            # convert camera params into rays
            query_rays_o, query_rays_d = poses_to_rays(query_poses, query_focals, *support_imgs.shape[-2:])

            # flatten rays and image pixels
            targets = einops.rearrange(query_imgs, "b n c h w -> b (n h w) c")
            query_rays_o = einops.rearrange(query_rays_o, "b n h w c -> b (n h w) c")
            query_rays_d = einops.rearrange(query_rays_d, "b n h w c -> b (n h w) c")

            if epoch % exp_config.test_all_rays_freq == 0 or not exp_config.subsample_during_eval:
                ray_subbatch_size_eval = self.config.experiment.ray_subbatch_size_eval
                outputs = model.module.forward_by_subbatch_ray(
                    support_imgs,
                    support_poses,
                    support_focals,
                    z_range=z_range,
                    query_rays_o=query_rays_o,
                    query_rays_d=query_rays_d,
                    ray_subbatch_size=ray_subbatch_size_eval,
                    is_train=False,
                )
            else:
                query_rays_o, query_rays_d, targets = self.ray_subsampler.subsample(
                    query_rays_o, query_rays_d, targets, epoch=epoch, is_train=False
                )

                # generate coordinate inputs
                coord, z_values = model.module.sample_coord_input(
                    query_rays_o, query_rays_d, z_range, augment=True, with_z_values=True
                )

                # keep_xs_shape = subsample_coord_idxs is None
                outputs = model(
                    support_imgs,
                    support_poses,
                    support_focals,
                    z_values=z_values,
                    coord=coord,
                    return_rgb=True,
                    is_train=True,
                )

            loss = model.module.compute_loss(outputs, targets, reduction="sum")

            metrics = dict(loss_total=loss["loss_total"], mse=loss["mse"], psnr=loss["psnr"])
            accm.update(metrics, count=outputs.shape[0], sync=True, distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)

        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["xs"] = xs

        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model_ema = self.model_ema
        total_step = len(self.loader_trn) * epoch
        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
        else:
            pbar = enumerate(self.loader_trn)

        model.train()
        for it, xs in pbar:
            model.zero_grad(set_to_none=True)
            xs = {key: value.to(self.device, non_blocking=True) for key, value in xs.items()}

            support_imgs, support_poses, support_focals = xs["support_imgs"], xs["support_poses"], xs["support_focals"]
            query_imgs, query_poses, query_focals = xs["query_imgs"], xs["query_poses"], xs["query_focals"]
            z_range = xs["z_range"]

            # convert camera params into rays
            query_rays_o, query_rays_d = poses_to_rays(query_poses, query_focals, *support_imgs.shape[-2:])

            # flatten rays and image pixels
            targets = einops.rearrange(query_imgs, "b n c h w -> b (n h w) c")
            query_rays_o = einops.rearrange(query_rays_o, "b n h w c -> b (n h w) c")
            query_rays_d = einops.rearrange(query_rays_d, "b n h w c -> b (n h w) c")

            # subsample rays for efficient training
            assert query_rays_d.shape == query_rays_o.shape
            assert targets.shape == query_rays_d.shape

            query_rays_o, query_rays_d, targets = self.ray_subsampler.subsample(
                query_rays_o, query_rays_d, targets, epoch=epoch, is_train=True
            )

            # generate coordinate inputs
            coord, z_values = model.module.sample_coord_input(
                query_rays_o, query_rays_d, z_range, augment=True, with_z_values=True
            )

            outputs = model(
                support_imgs,
                support_poses,
                support_focals,
                z_values=z_values,
                coord=coord,
                return_rgb=True,
                is_train=True,
            )

            loss = model.module.compute_loss(outputs, targets)
            loss["loss_total"].backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optimizer.max_gn)
            optimizer.step()
            scheduler.step()

            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(loss_total=loss["loss_total"], mse=loss["mse"], psnr=loss["psnr"])
            accm.update(metrics, count=1)
            total_step += 1

            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)

        summary = accm.get_summary()
        summary["xs"] = xs
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 0 and epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary["xs"], epoch=epoch, mode=mode)

        self.writer.add_scalar("loss/loss_total", summary["loss_total"], mode, epoch)
        self.writer.add_scalar("loss/mse", summary["mse"], mode, epoch)
        self.writer.add_scalar("loss/psnr", summary["psnr"], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, xs, epoch=0, mode="valid", is_train=False):
        num_viz = 4

        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        support_imgs, support_poses, support_focals = (
            xs["support_imgs"][:num_viz],
            xs["support_poses"][:num_viz],
            xs["support_focals"][:num_viz],
        )
        query_imgs, query_poses, query_focals = (
            xs["query_imgs"][:num_viz],
            xs["query_poses"][:num_viz],
            xs["query_focals"][:num_viz],
        )
        z_range = xs["z_range"][:num_viz]

        # convert camera params into rays
        support_rays_o, support_rays_d = poses_to_rays(support_poses, support_focals, *support_imgs.shape[-2:])
        query_rays_o, query_rays_d = poses_to_rays(query_poses, query_focals, *support_imgs.shape[-2:])

        num_support = support_rays_o.shape[1]

        rays_o = torch.cat([support_rays_o, query_rays_o], dim=1)
        rays_d = torch.cat([support_rays_d, query_rays_d], dim=1)

        num_view, H, W, C = rays_o.shape[-4:]

        rays_o = einops.rearrange(rays_o, "b n h w c -> b (n h w) c")
        rays_d = einops.rearrange(rays_d, "b n h w c -> b (n h w) c")

        ray_subbatch_size_eval = self.config.experiment.ray_subbatch_size_eval
        outputs = self.model.module.forward_by_subbatch_ray(
            support_imgs,
            support_poses,
            support_focals,
            z_range,
            rays_o,
            rays_d,
            ray_subbatch_size=ray_subbatch_size_eval,
        )

        real_imgs = torch.cat([support_imgs, query_imgs], dim=1)  # (B, N, C, H, W)
        recon_imgs = einops.rearrange(outputs, "b (n h w) c -> b n c h w", n=num_view, h=H, w=W)  # (B, N, C, H, W)
        recon_imgs = torch.clamp(recon_imgs, 0, 1)

        viz_list = []
        for i in range(0, num_viz, 2):
            viz_list.extend(real_imgs[i : i + 2])
            viz_list.extend(recon_imgs[i : i + 2])
        grid = torch.cat(viz_list, dim=0)  # (B*N, C, H, W)
        grid = torchvision.utils.make_grid(grid, nrow=(num_support + 1) * 2)
        self.writer.add_image(f"novel view synthesis", grid, mode, epoch)
