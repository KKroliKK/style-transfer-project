import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from time import time
import json

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from aiogram.types.input_file import InputFile
from io import BytesIO
import asyncio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESOLUTION = 2560


def image_loader(style, content, resolution=RESOLUTION):
    content = Image.open(content)
    style = Image.open(style)

    scale = max(content.size)
    scale = (int(content.size[1] * resolution / scale), 
             int(content.size[0] * resolution / scale))

    loader = transforms.Compose([
        transforms.Resize(scale),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
  
    content = loader(content).unsqueeze(0)
    style = loader(style).unsqueeze(0)
    output = content.clone()

    return style.to(device, torch.float), content.to(device,torch.float), output.to(device,torch.float)