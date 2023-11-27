import torch
from torch import nn
from torchvision.models import vgg16


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.layers = nn.ModuleList(features).eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        layers_of_interest = {3, 8, 15, 22}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layers_of_interest:
                results.append(x)

        return results


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x):
        return self.layers(x)


class ConvNormReLULayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ResNormReLULayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.branch = nn.Sequential(
            ConvNormReLULayer(in_channels, out_channels, kernel_size, 1),
            ConvNormReLULayer(out_channels, out_channels, kernel_size, 1),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.branch(x)
        x = self.activation(x)
        return x


class ConvTanhLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride), nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormReLULayer(3, 48, 9, 1),
            ConvNormReLULayer(48, 96, 3, 2),
            ConvNormReLULayer(96, 192, 3, 2),
            ResNormReLULayer(192, 192, 3),
            ResNormReLULayer(192, 192, 3),
            ResNormReLULayer(192, 192, 3),
            ResNormReLULayer(192, 192, 3),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormReLULayer(192, 96, 3, 1),
            nn.Upsample(scale_factor=2),
            ConvNormReLULayer(96, 48, 3, 1),
            ConvTanhLayer(48, 3, 9, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ReCoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
