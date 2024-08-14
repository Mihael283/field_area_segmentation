import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))

        x2 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))

        x3 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))

        x4 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))

        x5 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, (1, 1)))))
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x)))

        return self.dropout(x)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3, self).__init__()

        self.resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        self.resnet101 = nn.Sequential(*list(self.resnet101.children())[:-2])

        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self.conv_1x1 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnet101(x)
        x = self.aspp(x)
        x = self.conv_1x1(x)
        x = self.upsample(x)

        return x

