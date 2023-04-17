import math
import torch

import numpy as np
from torch.nn import functional as F

LOG_SCALE_MIN = -7


def compute_entropy(x, normalized=False):
    if not normalized:
        x /= np.sum(x)
    h = -np.sum(x * np.log(x + 1e-10))
    return h


def update_codebook_with_entropy(codebook, code):
    code_h, code_w = code.shape[1:]
    try:
        code = code.view(-1).cpu().numpy()
    except:
        code = code.view(-1).numpy()
    code, code_cnt = np.unique(code, return_counts=True)
    code_cnt = code_cnt.astype(np.float32) / (code_h * code_w)
    codebook[code] += code_cnt
    code_ent_ = compute_entropy(codebook)
    return codebook, code_ent_


def torch_compute_entropy(x, normalized=False):
    if not normalized:
        x = x / torch.sum(x, dim=-1, keepdim=True)
    h = -torch.sum(x * torch.log(x + 1e-10), dim=-1)
    return h


def to_one_hot(tensor, n, fill_with=1.0):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def log_sum_exp(x, axis=1):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    # TF ordering -> NCHW format
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis) + 1e-7)


def log_prob_from_logits(x, axis=1):
    """numerically stable log_softmax implementation that prevents overflow"""
    # TF ordering -> NCHW format
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True) + 1e-7)


def soft_target_cross_entropy(input, target, reduction="mean", label_smoothing=0.0):
    unif = torch.ones_like(target) / target.shape[-1]
    target = label_smoothing * unif + (1 - label_smoothing) * target
    loss = torch.sum(-target * log_prob_from_logits(input, axis=-1), dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError()


def dmol(self, logit, x, reduction="mean", channel_bit=8):
    """log-likelihood for mixture of discretized logistics,"""
    """ assumes the data has been rescaled to [-1,1] interval """
    """ Modified to match NCHW format """

    xs = x.shape
    ls = logit.shape
    n_classes = math.pow(2, self.channel_bit)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[1] / 10)
    logit_probs = logit[:, :nr_mix, :, :]
    logit = torch.reshape(logit[:, nr_mix:, :, :], [ls[0], xs[1], nr_mix * 3, ls[2], ls[3]])  # 3 for mean, scale, coef

    means = logit[:, :, :nr_mix, :, :]
    log_scales = torch.clamp(logit[:, :, nr_mix : 2 * nr_mix, :, :], min=LOG_SCALE_MIN)
    coeffs = torch.tanh(logit[:, :, 2 * nr_mix : 3 * nr_mix, :, :])

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels

    zero_pad = torch.zeros([xs[0], xs[1], nr_mix, xs[2], xs[3]], requires_grad=False).cuda()
    x = x.contiguous()
    x = x.unsqueeze(2) + zero_pad

    m1 = means[:, 0, :, :, :].unsqueeze(1)
    m2 = (means[:, 1, :, :, :] + coeffs[:, 0, :, :, :] * x[:, 0, :, :, :]).unsqueeze(1)
    m3 = (
        means[:, 2, :, :, :] + coeffs[:, 1, :, :, :] * x[:, 0, :, :, :] + coeffs[:, 2, :, :, :] * x[:, 1, :, :, :]
    ).unsqueeze(1)

    means = torch.cat((m1, m2, m3), dim=1)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1 / (n_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1 / (n_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)  # log prob. for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log prob. for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-5)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - np.log((n_classes - 1) / 2)
    )

    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=1) + log_prob_from_logits(logit_probs, axis=1)

    loss = -torch.sum(log_sum_exp(log_probs), dim=[1, 2])
    if reduction == "mean":
        loss /= np.prod(xs[1:])
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


@torch.no_grad()
def dmol_get_means(self, logit):
    # Pytorch ordering
    logit = logit.permute(0, 2, 3, 1)

    ls = [int(y) for y in logit.size()]
    nr_mix = int(ls[-1] / 10)
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = logit[:, :, :, :nr_mix]
    logit = logit[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    _, argmax = logit_probs.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # select logistic parameters

    means = logit[:, :, :, :, :nr_mix]
    means = torch.sum(means * sel, dim=4)

    coeffs = torch.sum(torch.tanh(logit[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, dim=4)

    x = means
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.0), max=1.0)
    x1 = torch.clamp(torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.0), max=1.0)
    x2 = torch.clamp(torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.0), max=1.0)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out
