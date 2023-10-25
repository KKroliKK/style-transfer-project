import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESOLUTION = 240


def image_loader(style, content, resolution=RESOLUTION):
    content = Image.open(content)
    style = Image.open(style)

    scale = max(content.size)
    scale = (int(content.size[1] * resolution / scale), 
             int(content.size[0] * resolution / scale))

    loader = transforms.Compose([
        transforms.Resize(scale),
        transforms.ToTensor()])
  
    content = loader(content).unsqueeze(0)
    style = loader(style).unsqueeze(0)
    output = content.clone()

    return style.to(device, torch.float), content.to(device,torch.float), output.to(device,torch.float)


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