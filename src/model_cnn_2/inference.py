import torch
from model_cnn_2.helpers import im_load, im_save
from model_cnn_2.network import NeuralNetwork

# set consts
NUM_STYLE = 8
content_file = "../../data/model_cnn_2/source/img.png"
model_path = "models/model_cnn_2/"
save_path = "../../data/model_cnn_2/inference/"
style_index = -1


def inference(content_file: str, model_path: str, save_path: str, style_index: dict or int):
    """
    Inference of the network

    :param content_file: path of the image
    :param model_path: path of the trained model
    :param save_path: path save image to
    :param style_index: if value is -1 -> get all styles for content_file image
                        if value from 0 to (NUM_STYLES - 1) -> get specific style
                        if value is dict -> mix style, where key is used style and value is level
    """
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path + "model.ckpt"))
    model.eval()

    content_image = im_load(content_file, im_size=256)

    # one of the given styles
    if type(style_index) is int:
        # for all styles
        if style_index == -1:
            style_code = torch.eye(NUM_STYLE).unsqueeze(-1)
            content_image = content_image.repeat(NUM_STYLE, 1, 1, 1)

        # for specific style
        elif style_index in range(NUM_STYLE):
            style_code = torch.zeros(1, NUM_STYLE, 1)
            style_code[:, style_index, :] = 1

        else:
            raise RuntimeError("Not expected style index")
    # mix styles
    elif type(style_index) is dict:
        style_code = torch.zeros(1, NUM_STYLE, 1)

        for key in style_index:
            if key in range(NUM_STYLE):
                style_code[:, key, :] = style_index[key]
            else:
                raise RuntimeError("Not expected style index")

    stylized_image = model(content_image, style_code)

    im_save(stylized_image, save_path + "new_images.jpg")



import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from aiogram.types.input_file import InputFile
from io import BytesIO

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path + "model.ckpt"))
model.eval()


async def image_loader(content, resolution=3840):
    content = Image.open(content)

    max_pixels = max(content.size)
    scale = resolution / max_pixels if resolution < max_pixels else 1
    scale = (int(content.size[1] * scale), int(content.size[0] * scale))

    loader = transforms.Compose([transforms.Resize(scale), transforms.ToTensor()])
    content = loader(content).unsqueeze(0)

    return content.to(device, torch.float)


async def call_cnn2(content, style_index):
    content_image = await image_loader(content)

    style_code = torch.zeros(1, NUM_STYLE, 1)

    for key in style_index:
        style_code[:, key, :] = style_index[key]

    result = model(content_image, style_code)

    buff = BytesIO()
    torchvision.utils.save_image(result, buff, "PNG")
    buff.seek(0)
    result = InputFile(buff)

    return result


if __name__ == "__main__":
    inference(content_file, model_path, save_path, style_index)
