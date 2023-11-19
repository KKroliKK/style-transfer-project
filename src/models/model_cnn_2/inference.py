import torch
from helpers import im_load, im_save
from network import NeuralNetwork

# set consts
NUM_STYLE = 8
content_file = "../../data/model_cnn_2/source/img.png"
model_path = "../../models/model_cnn_2/"
save_path = "../../data/model_cnn_2/inference/"
style_index = -1


def inference():
    """Inference of the network"""

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path + "model.ckpt"))
    model.eval()

    content_image = im_load(content_file, im_size=256)
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

    stylized_image = model(content_image, style_code)
    im_save(stylized_image, save_path + "new_images.jpg")


if __name__ == "__main__":
    inference()
