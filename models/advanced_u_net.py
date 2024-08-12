import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        return self.relu(self.conv(x) + residual)

class AdvancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AdvancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ResidualConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(512, 1024))

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.up_conv4 = ResidualConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.up_conv3 = ResidualConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.up_conv2 = ResidualConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.up_conv1 = ResidualConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # Deep supervision
        self.deep4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.deep3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.deep2 = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5)
        x4 = self.att4(g=x, x=x4)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv4(x)
        deep4 = self.deep4(x)

        x = self.up3(x)
        x3 = self.att3(g=x, x=x3)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)
        deep3 = self.deep3(x)

        x = self.up2(x)
        x2 = self.att2(g=x, x=x2)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)
        deep2 = self.deep2(x)

        x = self.up1(x)
        x1 = self.att1(g=x, x=x1)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv1(x)

        logits = self.outc(x)

        if self.training:
            return logits, deep2, deep3, deep4
        else:
            return logits

