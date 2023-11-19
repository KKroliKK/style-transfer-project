import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def im_load(
    path: str, im_size: int = None, crop_size: int = None, centre_crop: bool = False
) -> torch.Tensor:
    """
    Load image

    :param path: Path of the original image
    :param im_size: Image size
    :param crop_size: Crop size
    :param centre_crop: Do centre crop if True else random
    :return: transformed image
    """
    transformer = get_transforms(
        im_size=im_size, crop_size=crop_size, centre_crop=centre_crop
    )
    image = Image.open(path).convert("RGB")

    return transformer(image).unsqueeze(0)


def im_save(image, save_path):
    """
    Save image

    :param image: styled image
    :param save_path: Path of the styled image to save
    """
    denormalize = T.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / std for std in STD]
    )
    image = denormalize(torchvision.utils.make_grid(image)).clamp_(0.0, 1.0)
    torchvision.utils.save_image(image, save_path)


def get_transforms(
    im_size: int = None, crop_size: int = None, centre_crop: bool = False
) -> T.Compose:
    """
    Make the transformer

    :param im_size: Image size
    :param crop_size: Crop size
    :param centre_crop: Do centre crop if True else random
    :return: Transformer
    """

    transformer = []
    if im_size:
        transformer.append(T.Resize(im_size))
    if crop_size:
        if centre_crop:
            transformer.append(T.CenterCrop(crop_size))
        else:
            transformer.append(T.RandomCrop(crop_size))

    transformer.append(T.ToTensor())
    transformer.append(T.Normalize(mean=MEAN, std=STD))

    return T.Compose(transformer)


class ImageDataset(Dataset):
    def __init__(self, dir_path: Path):
        self.images = sorted(list(dir_path.glob("*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        return img, index


class DataProcessor:
    def __init__(
        self, im_size: int = 256, crop_size: int = 240, centre_crop: bool = False
    ):
        self.transforms = get_transforms(
            im_size=im_size, crop_size=crop_size, centre_crop=centre_crop
        )

    def __call__(self, batch):
        images, indexes = list(zip(*batch))
        inputs = torch.stack([self.transforms(image) for image in images])

        return inputs, indexes
