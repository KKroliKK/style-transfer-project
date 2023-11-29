import os

import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter

import model_reconet.custom_transforms
from model_reconet.dataset import FlyingThings3DDataset, MonkaaDataset
from model_reconet.losses import *
from model_reconet.network import ReCoNet, Vgg16
from model_reconet.utils import (
    RunningLossesContainer,
    gram_matrix,
    occlusion_mask_from_flow,
    postprocess_reconet,
    preprocess_for_reconet,
    preprocess_for_vgg,
    tensors_sum,
)

if __name__ == "__main__":
    running_losses = RunningLossesContainer()
    global_step = 0

    gpu_device = 0
    data_dir = "../../data/"
    style = "../../mosaic.jpg"
    output_file = "../../models/model_reconet/model.pth"

    lr = 1e-3
    epochs = 2

    alpha = 1e4
    beta = 1e5
    gamma = 1e-5
    lambda_f = 1e5
    lambda_o = 2e5

    with torch.cuda.device(gpu_device):
        transform = transforms.Compose(
            [
                custom_transforms.Resize(640, 360),
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.ToTensor(),
            ]
        )
        monkaa = MonkaaDataset(os.path.join(data_dir, "monkaa"), transform)
        flyingthings3d = FlyingThings3DDataset(
            os.path.join(data_dir, "flyingthings3d"), transform
        )
        dataset = monkaa + flyingthings3d
        batch_size = 2
        traindata = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            drop_last=True,
        )

        model = ReCoNet().cuda()
        vgg = Vgg16().cuda()

        with torch.no_grad():
            style = Image.open(style)
            style = transforms.ToTensor()(style).cuda()
            style = style.unsqueeze_(0)
            style_vgg_features = vgg(preprocess_for_vgg(style))
            style_gram_matrices = [gram_matrix(x) for x in style_vgg_features]
            del style, style_vgg_features

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        writer = SummaryWriter()

        n_epochs = epochs
        for epoch in range(n_epochs):
            for sample in traindata:
                optimizer.zero_grad()

                sample = {name: tensor.cuda() for name, tensor in sample.items()}

                occlusion_mask = occlusion_mask_from_flow(
                    sample["optical_flow"],
                    sample["reverse_optical_flow"],
                    sample["motion_boundaries"],
                )

                # Compute ReCoNet features and output

                reconet_input = preprocess_for_reconet(sample["frame"])
                feature_maps = model.encoder(reconet_input)
                output_frame = model.decoder(feature_maps)

                previous_reconet_input = preprocess_for_reconet(
                    sample["previous_frame"]
                )
                previous_feature_maps = model.encoder(previous_reconet_input)
                previous_output_frame = model.decoder(previous_feature_maps)

                # Compute VGG features

                vgg_input_frame = preprocess_for_vgg(sample["frame"])
                vgg_output_frame = preprocess_for_vgg(postprocess_reconet(output_frame))
                input_vgg_features = vgg(vgg_input_frame)
                output_vgg_features = vgg(vgg_output_frame)

                vgg_previous_input_frame = preprocess_for_vgg(sample["previous_frame"])
                vgg_previous_output_frame = preprocess_for_vgg(
                    postprocess_reconet(previous_output_frame)
                )
                previous_input_vgg_features = vgg(vgg_previous_input_frame)
                previous_output_vgg_features = vgg(vgg_previous_output_frame)

                # Compute losses
                losses = {
                    "content loss": tensors_sum(
                        [
                            alpha
                            * content_loss(
                                output_vgg_features[2], input_vgg_features[2]
                            ),
                            alpha
                            * content_loss(
                                previous_output_vgg_features[2],
                                previous_input_vgg_features[2],
                            ),
                        ]
                    ),
                    "style loss": tensors_sum(
                        [
                            beta * style_loss(output_vgg_features, style_gram_matrices),
                            beta
                            * style_loss(
                                previous_output_vgg_features, style_gram_matrices
                            ),
                        ]
                    ),
                    "total variation": tensors_sum(
                        [
                            gamma * total_variation(output_frame),
                            gamma * total_variation(previous_output_frame),
                        ]
                    ),
                    "feature temporal loss": lambda_f
                    * feature_temporal_loss(
                        feature_maps,
                        previous_feature_maps,
                        sample["reverse_optical_flow"],
                        occlusion_mask,
                    ),
                    "output temporal loss": lambda_o
                    * output_temporal_loss(
                        reconet_input,
                        previous_reconet_input,
                        output_frame,
                        previous_output_frame,
                        sample["reverse_optical_flow"],
                        occlusion_mask,
                    ),
                }

                training_loss = tensors_sum(list(losses.values()))
                losses["training loss"] = training_loss

                training_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_losses.update(losses)

                    last_iteration = (
                        global_step == len(dataset) // batch_size * n_epochs - 1
                    )
                    if global_step % 25 == 0 or last_iteration:
                        average_losses = running_losses.get()
                        for key, value in average_losses.items():
                            writer.add_scalar(key, value, global_step)

                        running_losses.reset()

                    if global_step % 100 == 0 or last_iteration:
                        styled_test_image = stylize_image(
                            Image.open("test_image.jpeg"), model
                        )
                        writer.add_image("test image", styled_test_image, global_step)

                        for i in range(0, len(dataset), len(dataset) // 4):
                            sample = dataset[i]
                            styled_train_image_1 = stylize_image(sample["frame"], model)
                            styled_train_image_2 = stylize_image(
                                sample["previous_frame"], model
                            )
                            grid = torchvision.utils.make_grid(
                                [styled_train_image_1, styled_train_image_2]
                            )
                            writer.add_image(f"train images {i}", grid, global_step)

                    global_step += 1

        torch.save(model.state_dict(), output_file)
        writer.close()
