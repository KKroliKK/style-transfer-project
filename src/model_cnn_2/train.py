from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from model_cnn_2.helpers import DataProcessor, ImageDataset
from model_cnn_2.losses import (get_content_loss, get_style_loss,
                                get_total_variation_loss)
from model_cnn_2.network import NeuralNetwork

# set nodes
content_nodes = ["relu_3_3"]
style_nodes = ["relu_1_2", "relu_2_2", "relu_3_3", "relu_4_2"]
return_nodes = {3: "relu_1_2", 8: "relu_2_2", 15: "relu_3_3", 22: "relu_4_2"}

# set consts
NUM_STYLE = 8
device = torch.device("cuda")
lr = 1e-3
batch_size = 8
style_weight = 5
tv_weight = 1e-5
iterations = 40000
style_path = "../../data/model_cnn_2/styles/"
content_path = "../../data/model_cnn_2/contents/"
model_path = "../../models/model_cnn_2/"


def get_dataloaders():
    """Get content and style dataloaders"""

    content_dataset = ImageDataset(dir_path=Path(content_path))
    style_dataset = ImageDataset(dir_path=Path(style_path))

    data_processor = DataProcessor(im_size=256, crop_size=240, centre_crop=False)
    content_dataloader = DataLoader(
        dataset=content_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_processor,
    )
    style_dataloader = DataLoader(
        dataset=style_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_processor,
    )

    return content_dataloader, style_dataloader


def train(model, optimizer, loss_network, content_dataloader, style_dataloader):
    """Train Loop"""

    losses = {"content": [], "style": [], "tv": [], "total": []}

    for i in tqdm(range(iterations)):
        content_images, _ = next(iter(content_dataloader))
        style_images, style_indices = next(iter(style_dataloader))

        # set style codes
        style_codes = torch.zeros(batch_size, NUM_STYLE, 1)
        for b, s in enumerate(style_indices):
            style_codes[b, s] = 1

        content_images = content_images.to(device)
        style_images = style_images.to(device)
        style_codes = style_codes.to(device)

        # get pred from our network
        output_images = model(content_images, style_codes)

        # get output from vgg
        content_features = loss_network(content_images)
        style_features = loss_network(style_images)
        output_features = loss_network(output_images)

        # get losses
        style_loss = get_style_loss(output_features, style_features, style_nodes)
        content_loss = get_content_loss(
            output_features, content_features, content_nodes
        )
        total_variation_loss = get_total_variation_loss(output_images)
        total_loss = (
            content_loss + style_loss * style_weight + total_variation_loss * tv_weight
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses["content"].append(content_loss.item())
        losses["style"].append(style_loss.item())
        losses["tv"].append(total_variation_loss.item())
        losses["total"].append(total_loss.item())

        # verbose
        if i % 100 == 0 and i != 0:
            log = f"iteration: {i}"
            for k, v in losses.items():
                avg = sum(v) / len(v)
                log += f"; {k}: {avg:1.4f}"
                losses = {"content": [], "style": [], "tv": [], "total": []}
            print(log)

        if i % 500 == 0 and i != 0:
            torch.save(model.state_dict(), model_path + f"model-{i}.ckpt")

    torch.save(model.state_dict(), model_path + "model.ckpt")


def run():
    """Run training process"""

    # data
    content_dataloader, style_dataloader = get_dataloaders()

    # loss network
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    for param in vgg.parameters():
        param.requires_grad = False
    loss_network = create_feature_extractor(vgg, return_nodes).to(device)

    # network
    model = NeuralNetwork()
    # model.load_state_dict(torch.load(model_path + 'model3-2500.ckpt'))
    model.train()
    model = model.to(device)

    # set optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # run train loop
    train(model, optimizer, loss_network, content_dataloader, style_dataloader)


if __name__ == "__main__":
    run()
