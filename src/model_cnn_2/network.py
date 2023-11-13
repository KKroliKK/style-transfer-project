import torch
import torch.nn as nn


class ConditionalInstanceNormalization(nn.Module):
    """Conditional Instance Normalization implementation"""

    def __init__(self, num_style: int, num_ch: int):
        """
        num_style: Number of styles
        num_ch: Number of Feature maps
        """
        
        super(ConditionalInstanceNormalization, self).__init__()

        self.normalize = nn.InstanceNorm2d(num_ch, affine=False)
        self.offset = nn.Parameter(0.01 * torch.randn(1, num_style, num_ch))
        self.scale = nn.Parameter(1 + 0.01 * torch.randn(1, num_style, num_ch))

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        :param x: Input tensor
        :param style: Style tensor
        """

        b, c, h, w = x.size()

        # normalization step
        x = self.normalize(x)

        # scale and offset step
        scale = torch.sum(self.scale * style, dim=1).view(b, c, 1, 1)
        offset = torch.sum(self.offset * style, dim=1).view(b, c, 1, 1)
        x = x * scale + offset

        return x.view(b, c, h, w)


class ConvolutionalLayer(nn.Module):
    """Convolution layer with Conditional Instance Normalization"""

    def __init__(self, num_style: int, in_fm: int, out_fm: int, stride: int, activation: str, kernel_size: int):
        """
        :param num_style: Number of styles
        :param in_fm: Number of input Feature maps
        :param out_fm: Number of output Feature maps
        :param stride: Stride
        :param activation: ReLU (relu) or Linear (linear) or Sigmoid (sigmoid)
        :param kernel_size: Kernel_size
        """
        
        super(ConvolutionalLayer, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_fm, out_fm, kernel_size, stride)
        self.norm = ConditionalInstanceNormalization(num_style, out_fm)

        if activation == "relu":
            self.activation = nn.ReLU()

        elif activation == "linear":
            self.activation = lambda x: x

        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        :param x: Input tensor
        :param style: Style tensor
        """

        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x, style)
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    """ResidualBlock"""

    def __init__(self, num_style: int, in_fm: int, out_fm: int):
        """
        :param num_style: Number of styles
        :param in_fm: Number of input Feature maps
        :param out_fm: Number of output Feature maps
        """
        
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvolutionalLayer(num_style, in_fm, out_fm, 1, "relu", 3)
        self.conv2 = ConvolutionalLayer(num_style, out_fm, out_fm, 1, "linear", 3)

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        :param x: Input tensor
        :param style: Style tensor
        """

        out = self.conv1(x, style)
        out = self.conv2(out, style)

        return x + out


class UpsampleBlock(nn.Module):
    """Upsample Block"""

    def __init__(self, num_style: int, in_fm: int, out_fm: int):
        """
        :param num_style: Number of styles
        :param in_fm: Number of input Feature maps
        :param out_fm: Number of output Feature maps
        """
        
        super(UpsampleBlock, self).__init__()
        self.conv = ConvolutionalLayer(num_style, in_fm, out_fm, 1, "relu", 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        :param x: Input tensor
        :param style: Style tensor
        """

        x = self.upsample(x)
        x = self.conv(x, style)

        return x


class NeuralNetwork(nn.Module):
    """Style Transfer Neural Network"""

    def __init__(self, num_style: int = 16):
        """
        :param num_style: Number of styles
        """
        
        super(NeuralNetwork, self).__init__()
        self.conv1 = ConvolutionalLayer(num_style, 3, 32, 1, 'relu', 9)
        self.conv2 = ConvolutionalLayer(num_style, 32, 64, 2, 'relu', 3)
        self.conv3 = ConvolutionalLayer(num_style, 64, 128, 2, 'relu', 3)

        self.residual1 = ResidualBlock(num_style, 128, 128)
        self.residual2 = ResidualBlock(num_style, 128, 128)
        self.residual3 = ResidualBlock(num_style, 128, 128)
        self.residual4 = ResidualBlock(num_style, 128, 128)
        self.residual5 = ResidualBlock(num_style, 128, 128)

        self.upsampling1 = UpsampleBlock(num_style, 128, 64)
        self.upsampling2 = UpsampleBlock(num_style, 64, 32)

        self.conv4 = ConvolutionalLayer(num_style, 32, 3, 1, 'sigmoid', 9)

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        """
        :param x: Input tensor
        :param style: Style tensor
        """

        x = self.conv1(x, style)
        x = self.conv2(x, style)
        x = self.conv3(x, style)

        x = self.residual1(x, style)
        x = self.residual2(x, style)
        x = self.residual3(x, style)
        x = self.residual4(x, style)
        x = self.residual5(x, style)

        x = self.upsampling1(x, style)
        x = self.upsampling2(x, style)

        x = self.conv4(x, style)

        return x
