import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv(x)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = DoubleConv(in_channels,out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):

        features = self.conv(x)

        pooled = self.pool(features)

        return features, pooled

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)

        self.conv = DoubleConv(out_channels * 2,out_channels)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([skip, x],dim=1)

        x = self.conv(x)

        return x
class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoders
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoders
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Final Layer
        self.final_conv = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):

        # Encoders
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoders
        d4 = self.dec4(b, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        # Output
        output = self.final_conv(d1)

        return output