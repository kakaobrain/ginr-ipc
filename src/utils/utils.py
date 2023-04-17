import os
import logging
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from easydict import EasyDict as edict
from .writer import Writer


def config_setup(args, distenv, result_path):
    if args.eval:
        config = yaml.load(open(os.path.join(args.result_path, "config.yaml")), Loader=yaml.FullLoader)
        config = config_init(config)
        if hasattr(args, "test_batch_size"):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, "seed"):
            config.seed = args.seed
    elif args.resume:
        config = yaml.load(open(os.path.join(os.path.dirname(args.result_path), "config.yaml")), Loader=yaml.FullLoader)
        config = config_init(config)
    else:
        config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
        config = config_init(config)
        config.seed = args.seed

        if hasattr(config.experiment, "total_batch_size"):
            t_batch_size = config.experiment.total_batch_size
            l_batch_size = config.experiment.batch_size
            assert t_batch_size % (l_batch_size * distenv.world_size) == 0
            config.optimizer.grad_accm_steps = int(t_batch_size / (l_batch_size * distenv.world_size))
        else:
            config.experiment.total_batch_size = config.experiment.batch_size * distenv.world_size
            config.optimizer.grad_accm_steps = 1

        if distenv.master:
            config.result_path = result_path
            yaml.dump(config, open(os.path.join(result_path, "config.yaml"), "w"))

    return config

def logger_setup(args):
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank > 0 or args.node_rank > 0:
        return None, None, None

    if args.eval:
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        result_path = os.path.join(args.result_path, "val", now)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "val.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname), logging.StreamHandler()],
        )
    elif args.resume:
        result_path = os.path.dirname(args.result_path)
        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "train.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname, mode="a"), logging.StreamHandler()],
        )
    else:
        now = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_cfg_name = os.path.splitext(args.model_config.split("/")[-1])[0]
        result_path = os.path.join(args.result_path, model_cfg_name, now)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        writer = Writer(result_path)
        log_fname = os.path.join(result_path, "train.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_fname), logging.StreamHandler()],
        )
    logger = logging.getLogger(__name__)
    logger.info(args)

    return logger, writer, result_path

def config_init(config):
    config = edict(config)

    def set_default_attr(cfg, attr, default):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    set_default_attr(config.dataset.transforms, "type", None)
    set_default_attr(config.arch.hparams, "loss_type", "mse")
    set_default_attr(config.arch, "ema", None)
    set_default_attr(config.optimizer, "max_gn", None)
    set_default_attr(config.optimizer.warmup, "start_from_zero", True if config.optimizer.warmup.epoch > 0 else False)
    set_default_attr(config.optimizer, "type", "adamW")
    set_default_attr(config.optimizer.warmup, "mode", "linear")
    set_default_attr(config.experiment, "test_freq", 10)
    set_default_attr(config.experiment, "amp", False)

    return config

def set_seed(seed=None):
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed