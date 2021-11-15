# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as f


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)), nn.ReLU()]

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, factor):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample' and factor == 1:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        elif up_mode == 'upsample' and not factor == 1:
            self.up = nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            factor=1,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        self.factor = factor if up_mode == 'upsample' else 1

        self.down_path.append(UNetConvBlock(in_channels, 2 ** wf, padding, batch_norm))
        for i in range(depth - 1):
            if not self.factor == 1 and i == depth - 2:
                self.down_path.append(
                    UNetConvBlock(2 ** (wf + i), 2 ** (wf + i + 1) // self.factor, padding, batch_norm)
                )
            else:
                self.down_path.append(
                    UNetConvBlock(2 ** (wf + i), 2 ** (wf + i + 1), padding, batch_norm)
                )

        for i in reversed(range(depth - 1)):
            if self.factor == 1 or i == 0:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i), up_mode, padding, batch_norm, self.factor)
                )
            else:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i) // self.factor, up_mode, padding, batch_norm,
                                self.factor)
                )

        self.last = nn.Conv2d(2 ** wf, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = f.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetBase(nn.Module):
    def __init__(
            self,
            in_channels=1,
            depth=5,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            factor=1,
            junction_point=1,
    ):
        """
        :param in_channels:
        :param depth:
        :param wf:
        :param up_mode:
        :param junction_point: 0から4
        """
        super(UNetBase, self).__init__()
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()
        self.factor = factor if up_mode == 'upsample' else 1

        self.down_path.append(UNetConvBlock(in_channels, 2 ** wf, padding, batch_norm))
        for i in range(depth - 1):
            if not self.factor == 1 and i == depth - 2:
                self.down_path.append(
                    UNetConvBlock(2 ** (wf + i), 2 ** (wf + i + 1) // self.factor, padding, batch_norm)
                )
            else:
                self.down_path.append(
                    UNetConvBlock(2 ** (wf + i), 2 ** (wf + i + 1), padding, batch_norm)
                )

        for i in reversed(range(junction_point, depth - 1)):
            if self.factor == 1 or i == 0:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i), up_mode, padding, batch_norm, self.factor)
                )
            else:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i) // self.factor, up_mode, padding, batch_norm,
                                self.factor)
                )

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = f.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return x, blocks


class UNetClassifier(nn.Module):
    def __init__(
            self,
            n_classes=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
            factor=1,
            junction_point=1,
    ):
        super(UNetClassifier, self).__init__()
        self.junction_point = junction_point
        self.up_path = nn.ModuleList()
        self.factor = factor if up_mode == 'upsample' else 1

        for i in range(junction_point):
            if self.factor == 1 or i == 0:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i), up_mode, padding, batch_norm, self.factor)
                )
            else:
                self.up_path.append(
                    UNetUpBlock(2 ** (wf + i + 1), 2 ** (wf + i) // self.factor, up_mode, padding, batch_norm,
                                self.factor)
                )
        self.last = nn.Conv2d(2 ** wf, n_classes, kernel_size=1)

    def forward(self, x):
        h, blocks = x
        for i in reversed(range(self.junction_point)):
            h = self.up_path[i](h, blocks[i])

        return self.last(h)
