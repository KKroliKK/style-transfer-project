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


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std