import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mp_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_size, out_size, up_mode='upconv'):
        super(Up, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Upsample(mode='nearest', scale_factor=2)
        self.conv = DoubleConv(in_size, out_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        """diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        x2 = f.pad(x2, [diff_x // 2, diff_x // 2, diff_y // 2, diff_y // 2])"""
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch, sig=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.Sigmoid()
        self.sig = sig

    def forward(self, x):
        x = self.conv(x)
        if self.sig:
            x = self.act(x)
        return x


class CoDetectionCNN(nn.Module):
    def __init__(self, n_channels, n_classes, up_mode='upconv'):
        super().__init__()
        self.inc = InConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, up_mode=up_mode)
        self.up2 = Up(512, 256, up_mode=up_mode)
        self.up3_t1 = Up(256, 128, up_mode=up_mode)
        self.up3_t2 = Up(256, 128, up_mode=up_mode)
        self.up4_t1 = Up(128, 64, up_mode=up_mode)
        self.up4_t2 = Up(128, 64, up_mode=up_mode)

        self.out_t1 = OutConv(64, n_classes)
        self.out_t2 = OutConv(64, n_classes)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3_t1 = self.down1(x2_t1)
        x3_t2 = self.down1(x2_t2)

        x4 = torch.cat([x3_t1, x3_t2], dim=1)

        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        x8 = self.up1(x7, x6)
        x9 = self.up2(x8, x5)

        x10_t1 = self.up3_t1(x9, x3_t1)
        x10_t2 = self.up3_t2(x9, x3_t2)

        x11_t1 = self.up4_t1(x10_t1, x2_t1)
        x11_t2 = self.up4_t2(x10_t2, x2_t2)

        pred_t1 = self.out_t1(x11_t1)
        pred_t2 = self.out_t2(x11_t2)

        return pred_t1, pred_t2


class CoDetectionCNNJ1(nn.Module):
    def __init__(self, n_channels, n_classes, up_mode='upconv'):
        super().__init__()
        self.inc = InConv(n_channels, 64)

        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, up_mode=up_mode)
        self.up2 = Up(512, 256, up_mode=up_mode)
        self.up3 = Up(256, 128, up_mode=up_mode)
        self.up4_t1 = Up(128, 64, up_mode=up_mode)
        self.up4_t2 = Up(128, 64, up_mode=up_mode)

        self.out_t1 = OutConv(64, n_classes)
        self.out_t2 = OutConv(64, n_classes)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3 = torch.cat([x2_t1, x2_t2], dim=1)

        x4 = self.down1(x3)
        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        x8 = self.up1(x7, x6)
        x9 = self.up2(x8, x5)
        x10 = self.up3(x9, x4)

        x11_t1 = self.up4_t1(x10, x2_t1)
        x11_t2 = self.up4_t2(x10, x2_t2)

        pred_t1 = self.out_t1(x11_t1)
        pred_t2 = self.out_t2(x11_t2)

        return pred_t1, pred_t2


class CoDetectionBase(nn.Module):
    def __init__(self, n_channels, up_mode='upconv'):
        super(CoDetectionBase, self).__init__()
        factor = 2 if up_mode == 'upsample' else 1
        self.inc = InConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, up_mode=up_mode)
        self.up2 = Up(512, 256 // factor, up_mode=up_mode)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3_t1 = self.down1(x2_t1)
        x3_t2 = self.down1(x2_t2)

        x4 = torch.cat([x3_t1, x3_t2], dim=1)

        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        x8 = self.up1(x7, x6)
        x9 = self.up2(x8, x5)

        return (x9, x3_t2, x2_t1), (x9, x3_t2, x2_t2)


class CoDetectionBaseJ1(nn.Module):
    def __init__(self, n_channels, up_mode='upconv'):
        super(CoDetectionBaseJ1, self).__init__()
        self.inc = InConv(n_channels, 64)

        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, up_mode=up_mode)
        self.up2 = Up(512, 256, up_mode=up_mode)
        self.up3 = Up(256, 128, up_mode=up_mode)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3 = torch.cat([x2_t1, x2_t2], dim=1)

        x4 = self.down1(x3)
        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        x8 = self.up1(x7, x6)
        x9 = self.up2(x8, x5)
        x10 = self.up3(x9, x4)

        return (x10, x2_t1), (x10, x2_t2)


class CoDetectionClassifier(nn.Module):
    def __init__(self, n_classes, up_mode='upconv'):
        super(CoDetectionClassifier, self).__init__()
        factor = 2 if up_mode == 'upsample' else 1
        self.up3 = Up(256, 128 // factor, up_mode=up_mode)
        self.up4 = Up(128, 64 // factor, up_mode=up_mode)
        self.out = OutConv(64 // factor, n_classes)

    def forward(self, x):
        x9, x3, x2 = x
        x10 = self.up3(x9, x3)
        x11 = self.up4(x10, x2)
        out = self.out(x11)
        return out
