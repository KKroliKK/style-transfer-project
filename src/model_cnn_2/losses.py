import torch
from torch.nn.functional import mse_loss


def get_content_loss(
    features: torch.Tensor, targets: torch.Tensor, nodes: list
) -> torch.Tensor:
    """
    Get Content Loss

    :param features: Features of pred image from loss network
    :param targets: Features of content image from loss network
    :param nodes: list of content nodes
    :return: Content Loss value
    """

    loss = 0
    for node in nodes:
        loss += mse_loss(features[node], targets[node])
    return loss


def get_style_loss(
    features: torch.Tensor, targets: torch.Tensor, nodes: list
) -> torch.Tensor:
    """
    Get Style Loss

    :param features: Features of pred image from loss network
    :param targets: Features of style image from loss network
    :param nodes: list of style nodes
    :return: Style Loss value
    """

    def get_gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """
        From feature to Gram matrix

        :param x: Some feature
        :return: Gram matrix
        """

        b, c, h, w = x.size()
        f = x.flatten(2)
        g = torch.bmm(f, f.transpose(1, 2))
        return g.div(h * w)

    loss = 0
    for node in nodes:
        loss += mse_loss(
            get_gram_matrix(features[node]), get_gram_matrix(targets[node])
        )
    return loss


def get_total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Get Total Variation Loss

    :param x: output_images
    :return: Total Variation Loss value
    """

    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )
    return loss
