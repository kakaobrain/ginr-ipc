import logging

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)


class SubSampler:
    r"""
    `SubSampler` is designed to randomly select the subset of coordinates
    to efficiently train the transformer that generates INRs of given data.
    In the training loop, `subcoord_idx` generates sub-coordinates according to the `subsampler_config`.
    Then, in the traning loop, `subsample_coords` and `subsample_xs` slices the subset of features
    according to the generated `subcoord_idx`.
    """

    def __init__(self, subsamper_config):
        self.config = subsamper_config
        if self.config.type is not None and self.config.ratio == 1.0:
            self.config.type = None

    def subsample_coords_idx(self, xs, epoch=None):
        if self.config.type is None:
            subcoord_idx = None
        elif self.config.type == "random":
            subcoord_idx = self.subsample_random_idx(xs, ratio=self.config.ratio)
        else:
            raise NotImplementedError
        return subcoord_idx

    def subsample_random_idx(self, xs, ratio=None):
        batch_size = xs.shape[0]
        spatial_dims = list(xs.shape[2:])

        subcoord_idx = []
        num_spatial_dims = np.prod(spatial_dims)
        num_subcoord = int(num_spatial_dims * ratio)
        for idx in range(batch_size):
            rand_idx = torch.randperm(num_spatial_dims, device=xs.device)
            rand_idx = rand_idx[:num_subcoord]
            subcoord_idx.append(rand_idx.unsqueeze(0))
        return torch.cat(subcoord_idx, dim=0)

    @staticmethod
    def subsample_coords(coords, subcoord_idx=None):
        if subcoord_idx is None:
            return coords

        batch_size = coords.shape[0]
        sub_coords = []
        coords = coords.view(batch_size, -1, coords.shape[-1])
        for idx in range(batch_size):
            sub_coords.append(coords[idx : idx + 1, subcoord_idx[idx]])
        sub_coords = torch.cat(sub_coords, dim=0)
        return sub_coords

    @staticmethod
    def subsample_xs(xs, subcoord_idx=None):
        if subcoord_idx is None:
            return xs

        batch_size = xs.shape[0]
        permute_idx_range = [i for i in range(2, xs.ndim)]  # note: xs is originally channel-fist data format
        xs = xs.permute(0, *permute_idx_range, 1)  # convert xs into channel last type

        xs = xs.reshape(batch_size, -1, xs.shape[-1])
        sub_xs = []
        for idx in range(batch_size):
            sub_xs.append(xs[idx : idx + 1, subcoord_idx[idx]])
        sub_xs = torch.cat(sub_xs, dim=0)
        return sub_xs


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_subsampler = SubSampler(self.config.loss.subsample)
        assert self.config.loss.coord_noise in [None, "shift", "coordwise"]

    def get_accm(self):
        accm = AccmStageINR(
            scalar_metric_names=("loss_total", "mse", "psnr"),
            device=self.device,
        )
        return accm

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xs in pbar:
            model.zero_grad()

            xs = xs.to(self.device)  # [B, C, *]
            coord_inputs = model.module.sample_coord_input(xs, device=xs.device)

            outputs = model(xs, coord_inputs)
            targets = xs.detach()
            loss = model.module.compute_loss(outputs, targets, reduction="sum")

            metrics = dict(loss_total=loss["loss_total"], mse=loss["mse"], psnr=loss["psnr"])
            accm.update(metrics, count=xs.shape[0], sync=True, distenv=self.distenv)

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
            xs = xs.to(self.device, non_blocking=True)

            coord_inputs = model.module.sample_coord_input(xs, device=xs.device)

            if self.config.loss.coord_noise == "coordwise":
                # add noises to the coordinates for avoid overfitting on training coordinates.
                (B, *shape, input_dim) = coord_inputs.shape
                unif_noises = torch.rand(B, *shape, input_dim, device=coord_inputs.device)
                len_coord_range = self.config.arch.coord_sampler.coord_range[1] - self.config.arch.coord_sampler.coord_range[0]
                coord_noises = (unif_noises - 0.5) * len_coord_range / shape[0]
                coord_inputs = coord_inputs + coord_noises

            subsample_coord_idxs = self.coord_subsampler.subsample_coords_idx(xs, epoch=epoch)
            coord_inputs = self.coord_subsampler.subsample_coords(coord_inputs, subsample_coord_idxs)

            keep_xs_shape = subsample_coord_idxs is None
            outputs = model(xs, coord_inputs, keep_xs_shape=keep_xs_shape)

            targets = xs.detach()
            targets = self.coord_subsampler.subsample_xs(targets, subsample_coord_idxs)
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
        if epoch % 10 == 1 or epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary["xs"], upsample_ratio=1, epoch=epoch, mode=mode)
            self.reconstruct(summary["xs"], upsample_ratio=3, epoch=epoch, mode=mode)

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
    def reconstruct(self, xs, upsample_ratio=1, epoch=0, mode="valid"):
        r"""Reconstruct the input data according to `upsample_ratio` and logs the results
        Args
            xs (torch.Tensor) : the data to be reconstructed.
            upsample_ratio (int, float) : upsamling ratio in (0, \inf) for data ireconstruction.
                If `upsample_ratio<1` the reconstructed results will be down-sampled.
                If `upsample_ratio==1`, the reconstructed data have the same resolution with the input data `xs`.
                If `upsample_ratio>1` the reconstructed results have higher resolution than input data using coordinate interpolation of INRs.
            epoch (int) : the number of epoch to be logged.
            mode (str) : the prefix for logging the result (e.g. "valid, "train")
        """

        def get_recon_imgs(xs_real, xs_recon, upsample_ratio=1):
            xs_real = xs_real
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(xs_real, scale_factor=upsample_ratio)
            xs_recon = xs_recon
            xs_recon = torch.clamp(xs_recon, 0, 1)
            return xs_real, xs_recon

        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        assert upsample_ratio > 0

        xs_real = xs[:4]
        coord_inputs = model.module.sample_coord_input(xs_real, upsample_ratio=upsample_ratio, device=xs.device)

        xs_recon = model(xs_real, coord_inputs)

        if self.config.arch.coord_sampler.data_type == "audio":
            xs_real, xs_recon = xs_real, xs_recon
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(xs_real, scale_factor=upsample_ratio)
            sampling_rate = self.config.dataset.transforms.sampling_rate * upsample_ratio
            for i in range(len(xs_real)):
                self.writer.add_audio(
                    f"reconstruction_x{upsample_ratio}/real_{i+1}",
                    xs_real[i].squeeze(0),
                    mode,
                    sampling_rate=sampling_rate,
                    epoch=epoch,
                )
                self.writer.add_audio(
                    f"reconstruction_x{upsample_ratio}/recon_{i+1}",
                    xs_recon[i].squeeze(0),
                    mode,
                    sampling_rate=sampling_rate,
                    epoch=epoch,
                )
        elif self.config.arch.coord_sampler.data_type == "image":
            xs_real, xs_recon = get_recon_imgs(xs_real, xs_recon, upsample_ratio)
            grid = torch.cat([xs_real, xs_recon], dim=0)
            grid = torchvision.utils.make_grid(grid, nrow=4)
            self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
