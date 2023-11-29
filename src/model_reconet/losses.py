import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from model_reconet.utils import (
    gram_matrix,
    l2_squared,
    postprocess_reconet,
    preprocess_for_reconet,
    resize_optical_flow,
    rgb_to_luminance,
    warp_optical_flow,
)


def output_temporal_loss(
    input_frame,
    previous_input_frame,
    output_frame,
    previous_output_frame,
    reverse_optical_flow,
    occlusion_mask,
):
    input_diff = input_frame - warp_optical_flow(
        previous_input_frame, reverse_optical_flow
    )
    output_diff = output_frame - warp_optical_flow(
        previous_output_frame, reverse_optical_flow
    )
    luminance_input_diff = rgb_to_luminance(input_diff).unsqueeze_(1)

    n, c, h, w = input_frame.shape
    loss = l2_squared(occlusion_mask * (output_diff - luminance_input_diff)) / (h * w)
    return loss


def feature_temporal_loss(
    feature_maps, previous_feature_maps, reverse_optical_flow, occlusion_mask
):
    n, c, h, w = feature_maps.shape

    reverse_optical_flow_resized = resize_optical_flow(reverse_optical_flow, h, w)
    occlusion_mask_resized = torch.nn.functional.interpolate(
        occlusion_mask, size=(h, w), mode="nearest"
    )

    feature_maps_diff = feature_maps - warp_optical_flow(
        previous_feature_maps, reverse_optical_flow_resized
    )
    loss = l2_squared(occlusion_mask_resized * feature_maps_diff) / (c * h * w)

    return loss


def content_loss(content_feature_maps, style_feature_maps):
    n, c, h, w = content_feature_maps.shape

    return l2_squared(content_feature_maps - style_feature_maps) / (c * h * w)


def style_loss(content_feature_maps, style_gram_matrices):
    loss = 0
    for content_fm, style_gm in zip(content_feature_maps, style_gram_matrices):
        loss += l2_squared(gram_matrix(content_fm) - style_gm)
    return loss


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
        torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    )


def stylize_image(image, model):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    image = image.cuda().unsqueeze_(0)
    image = preprocess_for_reconet(image)
    styled_image = model(image).squeeze()
    styled_image = postprocess_reconet(styled_image)
    return styled_image
