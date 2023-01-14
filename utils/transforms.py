import torchvision.transforms as T
from utils.colors import color_shift_from_targets, color_shift
import albumentations as A
import numpy as np
import kornia
import torch


def random_erode_dilate(x):
    erode = np.random.choice([True, False])
    if erode:
        return kornia.morphology.dilation(x.unsqueeze(0), kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)
    else:
        return kornia.morphology.erosion(x.unsqueeze(0), kernel=torch.ones(np.random.choice([3,4]), np.random.choice([2,3]))).squeeze(0)


TRANSFORM_DICT = {
    "default":
        T.Compose([
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.5),
            T.RandomInvert(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.ToPILImage(),
        ]),
    "album":
        T.Compose([
            lambda x: A.PixelDropout(dropout_prob=0.01, drop_value=0, p=0.10)(image=np.array(x))["image"],
            lambda x: A.GaussNoise(var_limit=(10.0, 100.0), mean=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.ImageCompression(quality_lower=0, quality_upper=50, p=0.20)(image=np.array(x))["image"],
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(15, sigma=(1, 3))], p=0.5),
            T.RandomInvert(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([random_erode_dilate], p=0.25),
            T.ToPILImage(),
        ]),
    "pr":
        T.Compose([
            T.ToTensor(),
            lambda x: color_shift_from_targets(x, targets=[[234,234,212], [225, 207, 171]]),
            T.RandomApply([T.GaussianBlur(11)], p=0.35),
            T.ToPILImage()
        ]),
    "trdg":
        T.Compose([
            T.ToTensor(),
            T.RandomGrayscale(p=1.0),
            T.RandomApply([random_erode_dilate], p=0.6),
            T.RandomApply([T.GaussianBlur(9, sigma=(1, 2))], p=0.5),
            T.ToPILImage(),
            lambda x: A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.ImageCompression(quality_lower=0, quality_upper=100, p=0.20)(image=np.array(x))["image"],
            T.ToPILImage(),
        ]),
    "trdgcolor":
        T.Compose([
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomGrayscale(p=0.25),
            T.RandomApply([random_erode_dilate], p=0.6),
            T.RandomApply([T.GaussianBlur(9, sigma=(1, 2))], p=0.5),
            T.ToPILImage(),
            lambda x: A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.ImageCompression(quality_lower=0, quality_upper=100, p=0.20)(image=np.array(x))["image"],
            T.ToPILImage(),
        ]),
    "simple":
        T.Compose([
            T.ToTensor(),
            T.RandomGrayscale(p=1.0),
            T.ToPILImage(),
            lambda x: A.GaussNoise(var_limit=(10.0, 150.0), mean=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.ImageCompression(quality_lower=0, quality_upper=100, p=0.20)(image=np.array(x))["image"],
            T.ToPILImage(),
        ]),
}
