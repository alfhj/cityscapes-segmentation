import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class TransConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, output_padding),
            #nn.BatchNorm2d(output_dim),
            #nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, retain=True):
        super().__init__()

        self.conv1 = ConvBlock(input_dim, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        self.neck = ConvBlock(256, 512, 3, 1)

        self.upconv4 = TransConvBlock(512, 256)
        self.dconv4 = ConvBlock(512, 256)
        self.upconv3 = TransConvBlock(256, 128)
        self.dconv3 = ConvBlock(256, 128)
        self.upconv2 = TransConvBlock(128, 64)
        self.dconv2 = ConvBlock(128, 64)
        self.upconv1 = TransConvBlock(64, 32)
        self.dconv1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, output_dim, 3, 1, 1)
        self.retain = retain

    def forward(self, x):
        # Encoder Network

        # Conv down 1
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2, stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2, stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2, stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2, stride=2)

        # Bottleneck
        neck = self.neck(pool4)

        # Decoder Network

        # Upconv 1
        upconv4 = self.upconv4(neck)
        # Skip connection 1
        dconv4 = self.dconv4(torch.cat([upconv4, conv4], dim=1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        # Skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3, conv3], dim=1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        # Skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2, conv2], dim=1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        # Skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1, conv1], dim=1))
        # Output Layer
        out = self.out(dconv1)

        if self.retain == True:
            out = F.interpolate(out, list(x.shape)[2:])

        return out
