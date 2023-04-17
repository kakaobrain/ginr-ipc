import os
import imageio
import einops
import json
from pathlib import Path
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchaudio

from .base import ImageFolder


class ImageNette(Dataset):
    """Dataset for ImageNette that contains 10 classes of ImageNet.
    Dataset parses the pathes of images and load the image using PIL loader.

    Args:
        split: "train" or "val"
        transform (sequence or torch.nn.Module): list of transformations
    """

    root = Path(__file__).parent.parent.parent.joinpath("data/imagenette")

    def __init__(self, split="train", transform=None):
        assert split in ["train", "val"]
        self.transform = transform

        root_path = os.path.join(ImageNette.root, split)
        class_folders = sorted(os.listdir(root_path))
        self.data_path = []

        for class_folder in class_folders:
            filenames = sorted(os.listdir(os.path.join(root_path, class_folder)))
            for name in filenames:
                self.data_path.append(os.path.join(root_path, class_folder, name))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of data_path
        Returns:
            img (torch.Tensor): (C, H, W) shape of tensor
        """
        img = Image.open(self.data_path[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageOnlyDataset(Dataset):
    """
    Dataset wrapper which only returns images (not targets).
    We assume that `dataset.__getitem__` returns the image as the first element.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        value = self.dataset[item]
        if isinstance(value, tuple):
            return value[0]
        else:
            return value


class FFHQ(ImageFolder):
    root = Path(__file__).parent.parent.parent.joinpath("data/FFHQ")
    train_list_file = Path(__file__).parent.joinpath("assets/ffhqtrain.txt")
    val_list_file = Path(__file__).parent.joinpath("assets/ffhqvalidation.txt")

    def __init__(self, split="train", **kwargs):
        super().__init__(FFHQ.root, FFHQ.train_list_file, FFHQ.val_list_file, split, **kwargs)


class Celeba(Dataset):
    root = Path(__file__).parent.parent.parent.joinpath("data/CelebA_aligned")

    def __init__(self, split, transform=None):
        self.transform = transform

        if split == "train":
            s, t = 1, 162770
        elif split == "val":
            s, t = 162771, 182637
        elif split == "test":
            s, t = 182638, 202599
        else:
            raise ValueError("split not in 'train', 'val' or 'test'")

        self.data = []
        for i in range(s, t + 1):
            path = os.path.join(self.root, f"{i:06}.jpg")
            self.data.append(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img


class LibriSpeech(torchaudio.datasets.LIBRISPEECH):
    """LIBRISPEECH dataset after removing labels."""
    root = Path(__file__).parent.parent.parent.joinpath("data/LibriSpeech/original")

    def __init__(self, url="train-clean-100", download=True, transform=None):
        super().__init__(LibriSpeech.root, url=url, download=download)
        self.transform = transform

    def __getitem__(self, index):
        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        _datapoint = super().__getitem__(index)[0].float()
        _datapoint = _datapoint.unsqueeze(-2)  # audio has 1 channel dimension
        if self.transform is None:
            return _datapoint
        else:
            return self.transform(_datapoint)


class LearnitShapenet(Dataset):
    # Reference: https://github.com/tancik/learnit/blob/main/Experiments/shapenet.ipynb

    root = Path(__file__).parent.parent.parent.joinpath("data/learnit_shapenet")

    def __init__(self, category, config):
        root = LearnitShapenet.root
        assert category in ["cars", "chairs", "lamps"]

        self.split = config.split
        assert self.split in ["train", "test"]

        self.n_support = config.n_support
        self.n_query = config.n_query
        self.repeat = config.repeat
        self.views_range = config.views_range

        with open(os.path.join(root, category[: -len("s")] + "_splits.json"), "r") as f:
            obj_ids = json.load(f)[self.split]
        _data = [os.path.join(root, category, _) for _ in obj_ids]

        self.data = []
        for x in _data:
            if os.path.exists(os.path.join(x, "transforms.json")):
                self.data.append(x)
            else:
                print(f"Missing obj at {x}, skipped.")

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.data)

        train_ex_dir = self.data[idx]
        with open(os.path.join(train_ex_dir, "transforms.json"), "r") as fp:
            meta = json.load(fp)
        camera_angle_x = float(meta["camera_angle_x"])
        frames = meta["frames"]

        if self.views_range is not None and self.split == "train":
            frames = frames[self.views_range[0] : self.views_range[1]]

        frames = np.random.choice(frames, self.n_support + self.n_query, replace=False)

        imgs = []
        poses = []
        for frame in frames:
            fname = os.path.join(train_ex_dir, os.path.basename(frame["file_path"]) + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        H, W = imgs[0].shape[:2]
        assert H == W

        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
        poses = np.array(poses).astype(np.float32)

        imgs = einops.rearrange(torch.from_numpy(imgs), "n h w c -> n c h w")
        poses = torch.from_numpy(poses)[:, :3, :4]
        focal = torch.ones(len(poses), 2) * float(focal)
        t = self.n_support

        z_range = torch.from_numpy(np.array([2, 6]))
        return {
            "support_imgs": imgs[:t],
            "support_poses": poses[:t],
            "support_focals": focal[:t],
            "query_imgs": imgs[t:],
            "query_poses": poses[t:],
            "query_focals": focal[t:],
            "z_range": z_range,
        }
