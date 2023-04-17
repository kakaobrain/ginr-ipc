import os

import torch

from .mydatasets import ImageNette, FFHQ, ImageOnlyDataset, LearnitShapenet, LibriSpeech, Celeba
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split="train", is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split="val", is_eval=is_eval)

    if config.dataset.type == "imagenette":
        dataset_trn = ImageNette(split="train", transform=transforms_trn)
        dataset_val = ImageNette(split="val", transform=transforms_val)
    elif config.dataset.type == "ffhq":
        dataset_trn = FFHQ(split="train", transform=transforms_trn)
        dataset_val = FFHQ(split="val", transform=transforms_val)
    elif config.dataset.type == "celeba":
        dataset_trn = Celeba(split="train", transform=transforms_trn)
        dataset_val = Celeba(split="test", transform=transforms_val)
    elif config.dataset.type == "librispeech":
        dataset_trn = LibriSpeech("train-clean-100", transform=transforms_trn)
        dataset_val = LibriSpeech("test-clean", transform=transforms_val)
    elif config.dataset.type in ["LearnitShapenet-cars", "LearnitShapenet-chairs", "LearnitShapenet-lamps"]:
        category_name = config.dataset.type.split("-")[-1]
        dataset_trn = LearnitShapenet(category_name, config=config.dataset.train_config)
        dataset_val = LearnitShapenet(category_name, config=config.dataset.val_config)
    else:
        raise ValueError("%s not supported..." % config.dataset.type)

    if config.get("trainer", "") == "stage_inr":
        dataset_trn = ImageOnlyDataset(dataset_trn)
        dataset_val = ImageOnlyDataset(dataset_val)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f"#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}")

    return dataset_trn, dataset_val
