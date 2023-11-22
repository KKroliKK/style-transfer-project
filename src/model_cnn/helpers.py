import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(style, content, resolution):
    content = Image.open(content)
    style = Image.open(style)

    max_pixels = max(content.size)
    scale = resolution / max_pixels if resolution < max_pixels else 1
    scale = (int(content.size[1] * scale), int(content.size[0] * scale))

    loader = transforms.Compose([transforms.Resize(scale), transforms.ToTensor()])

    content = loader(content).unsqueeze(0)
    style = loader(style).unsqueeze(0)
    output = content.clone()

    return (
        style.to(device, torch.float),
        content.to(device, torch.float),
        output.to(device, torch.float),
    )


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    plt.ion()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
