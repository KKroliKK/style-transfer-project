import json
from io import BytesIO

import torch
import torch.optim as optim
import torchvision
from aiogram.types.input_file import InputFile

from model_cnn.create_model import cnn
from model_cnn.helpers import image_loader
from model_cnn.image_transformations import (cnn_normalization_mean,
                                             cnn_normalization_std)
from model_cnn.run import run_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


async def get_transformed_photo(
    style, content, content_weight=30, optimizer=optim.LBFGS
):
    with open("data/resolution.json") as json_file:
        data = json.load(json_file)
        resolution = data["resolution"]

    style_img, content_img, input_img = image_loader(style, content, resolution)

    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    result = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        content_layers,
        style_layers,
        content_weight=content_weight,
        optimizer=optimizer,
    )

    buff = BytesIO()
    torchvision.utils.save_image(result, buff, "PNG")
    buff.seek(0)
    result = InputFile(buff)

    return result


def transfer_style(style, content, input, content_weight=30):
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    result = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content,
        style,
        input,
        content_layers,
        style_layers,
        content_weight=content_weight,
    )

    return result
