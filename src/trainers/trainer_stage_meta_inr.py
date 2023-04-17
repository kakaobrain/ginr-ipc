import logging

import torch
import torchvision
from tqdm import tqdm

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accm(self):
        n_inner_step = self.config.arch.n_inner_step
        accm = AccmStageINR(
            scalar_metric_names=("loss_total", "mse", "psnr"),
            vector_metric_names=("inner_mse", "inner_psnr"),
            vector_metric_lengths=(n_inner_step, n_inner_step),
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

            outputs, _, collated_history = model(xs, coord_inputs, is_training=False)
            targets = xs.detach()
            loss = model.module.compute_loss(outputs, targets, reduction="sum")

            metrics = dict(
                loss_total=loss["loss_total"],
                mse=loss["mse"],
                psnr=loss["psnr"],
                inner_mse=collated_history["mse"],
                inner_psnr=collated_history["psnr"],
            )
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
            outputs, _, collated_history = model(xs, coord=coord_inputs, is_training=True)

            targets = xs.detach()
            loss = model.module.compute_loss(outputs, targets)
            loss["loss_total"].backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.optimizer.max_gn)
            optimizer.step()
            scheduler.step()

            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(
                loss_total=loss["loss_total"],
                mse=loss["mse"],
                psnr=loss["psnr"],
                inner_mse=collated_history["mse"] / xs.shape[0],
                inner_psnr=collated_history["psnr"] / xs.shape[0],
            )
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

        xs_recon, _, collated_history = model(xs_real, coord_inputs, is_training=False)
        xs_real, xs_recon = get_recon_imgs(xs_real, xs_recon, upsample_ratio)

        grid = []
        if upsample_ratio == 1:
            inner_step_recons = collated_history["recons"].clamp(0, 1)
            grid.append(inner_step_recons)
        grid.extend([xs_recon.unsqueeze(1), xs_real.unsqueeze(1)])

        grid = torch.cat(grid, dim=1)
        nrow = grid.shape[1]
        grid = grid.reshape(-1, *xs_recon.shape[1:])
        grid = torchvision.utils.make_grid(grid, nrow=nrow)
        self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
