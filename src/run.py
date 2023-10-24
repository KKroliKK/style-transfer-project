import torch
from time import time
import torchvision.models as models
import asyncio

from create_model import get_style_model_and_losses, get_input_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESOLUTION = 2560


cnn = models.vgg19(pretrained=True).features.to(device).eval()


async def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, 
                       content_layers, style_layers, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""

    tick = time()

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img,
        content_layers, style_layers)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]

    executing = time()

    while run[0] <= num_steps:
        
        # Allow other users to interact with Bot while
        # photo processing
        if (time() - executing) > 1.0:
            await asyncio.sleep(0.1)
            executing = time()

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)
    
    print(round((time() - tick), 2))
    return input_img