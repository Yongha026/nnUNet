import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm2d(out_ch, eps=1e-05, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            nn.InstanceNorm2d(out_ch, eps=1e-05, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Recommended_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder blocks matching features_per_stage: [32, 64, 128, 256, 512, 512]
        self.enc0 = ConvBlock(input_channels, 32, stride=(1, 1))
        self.enc1 = ConvBlock(32, 64, stride=(2, 2))
        self.enc2 = ConvBlock(64, 128, stride=(2, 2))
        self.enc3 = ConvBlock(128, 256, stride=(2, 2))
        self.enc4 = ConvBlock(256, 512, stride=(2, 2))
        self.enc5 = ConvBlock(512, 512, stride=(2, 2))  # Bottleneck

        # Upsampling blocks using ConvTranspose2d
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=False)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)

        # Decoder blocks
        self.dec4 = ConvBlock(1024, 512)
        self.dec3 = ConvBlock(512, 256)
        self.dec2 = ConvBlock(256, 128)
        self.dec1 = ConvBlock(128, 64)
        self.dec0 = ConvBlock(64, 32)

        # Segmentation projection heads
        self.seg0 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        if self.deep_supervision:
            self.seg1 = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
            self.seg2 = nn.Conv2d(128, num_classes, kernel_size=1, bias=False)
            self.seg3 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
            self.seg4 = nn.Conv2d(512, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Encoder path
        x0 = self.enc0(x)  # out size: (B, 32, H, W)
        x1 = self.enc1(x0)  # out size: (B, 64, H/2, W/2)
        x2 = self.enc2(x1)  # out size: (B, 128, H/4, W/4)
        x3 = self.enc3(x2)  # out size: (B, 256, H/8, W/8)
        x4 = self.enc4(x3)  # out size: (B, 512, H/16, W/16)
        x5 = self.enc5(x4)  # out size: (B, 512, H/32, W/32)

        # Decoder path with skip connection concatenations
        d4 = self.dec4(torch.cat([self.up4(x5), x4], dim=1))  # (B, 512, H/16, W/16)
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))  # (B, 256, H/8, W/8)
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))  # (B, 128, H/4, W/4)
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))  # (B, 64, H/2, W/2)
        d0 = self.dec0(torch.cat([self.up0(d1), x0], dim=1))  # (B, 32, H, W)

        # Output predictions
        out0 = self.seg0(d0)

        if self.training and self.deep_supervision:
            out1 = self.seg1(d1)
            out2 = self.seg2(d2)
            out3 = self.seg3(d3)
            out4 = self.seg4(d4)
            # Returns outputs from highest to lowest resolution for deep supervision
            return [out0, out1, out2, out3, out4]

        return out0
