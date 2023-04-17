import random
import torchvision.transforms as transforms


class CropAudio(object):
    def __init__(self, total_secs=10, patch_secs=-1, sampling_rate=16000, normalize=True, random_crop=False):
        self.total_secs = total_secs
        self.sampling_rate = sampling_rate
        self.num_waveform_samples = int(total_secs * sampling_rate)
        self.normalize = normalize
        self.random_crop = random_crop

        if patch_secs == -1 and not random_crop:
            assert not self.total_secs == -1
            self.patch_secs = self.total_secs
        else:
            # assert 0 < patch_secs <= total_secs
            self.patch_secs = patch_secs
        self.num_patch_samples = int(self.patch_secs * sampling_rate)

    def __call__(self, xs):
        _audio = xs

        if self.normalize:
            _audio = (_audio + 1.0) / 2.0

        _audio = _audio[..., : self.num_waveform_samples]  # trimeed overlengthed parts of audios

        if self.random_crop:
            if self.num_patch_samples == self.num_waveform_samples:
                pass
            else:
                # assert self.num_patch_samples < self.num_waveform_samples
                max_start_idx = max(_audio.shape[-1] - self.num_patch_samples, 0)
                start_idx = random.randint(0, max_start_idx)
                _audio = _audio[..., start_idx : start_idx + self.num_patch_samples]
        else:
            _audio = _audio[..., : self.num_patch_samples]

        # If the audio is shorter than num_patch_samples,
        # the remain parts are padded to keep the length of audios in the batch same
        audio = _audio.new_zeros(_audio.shape[0], self.num_patch_samples)
        if self.normalize:
            audio = audio + 0.5
        audio[..., : _audio.shape[-1]] = _audio
        return audio


def create_transforms(config, split="train", is_eval=False):
    if config.transforms.type in ["imagenette178x178", "imagenette256x256", "imagenette512x512"]:
        resolution = int(config.transforms.type.split("x")[-1])
        if split == "train" and not is_eval:
            transforms_ = [
                transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
    elif config.transforms.type in ["celeba178x178"]:
        resolution = int(config.transforms.type.split("x")[-1])
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    elif config.transforms.type in ["ffhq178x178", "ffhq256x256", "ffhq512x512", "ffhq1024x1024"]:
        resolution = int(config.transforms.type.split("_")[0].split("x")[-1])
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    elif config.transforms.type in ["librispeech"]:
        if "train" in split and not is_eval:
            transforms_ = [
                CropAudio(
                    config.transforms.total_secs,
                    config.transforms.patch_secs,
                    config.transforms.sampling_rate,
                    config.transforms.normalize,
                    config.transforms.random_crop,
                ),
            ]
        else:
            transforms_ = [
                CropAudio(
                    config.transforms.total_secs,
                    config.transforms.patch_secs,
                    config.transforms.sampling_rate,
                    config.transforms.normalize,
                    random_crop=False,
                ),
            ]
    else:
        raise NotImplementedError("%s not implemented.." % config.transforms.type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_
