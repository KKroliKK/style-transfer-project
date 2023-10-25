import torch
import json
import torchvision

from aiogram.types.input_file import InputFile
from io import BytesIO

from src.model_cnn.helpers import image_loader
from src.model_cnn.run import run_style_transfer
from src.model_cnn.image_transformations import cnn_normalization_mean, cnn_normalization_std
from src.model_cnn.create_model import cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transformed_photo(style, content, content_weight=30):

    with open("style_transfer/resolution.json") as json_file:
        data = json.load(json_file)
        resolution = data['resolution']

    style_img, content_img, input_img = image_loader(style, content, resolution)
    
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    result = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, content_layers, style_layers,
                                content_weight=content_weight)    

    buff = BytesIO()
    torchvision.utils.save_image(result, buff, 'PNG')
    buff.seek(0)
    result = InputFile(buff)

    return result


def transfer_style(style, content, input):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    result = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content, style, input, content_layers, style_layers,
                                content_weight=30)

    return result    