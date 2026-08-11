import torch
import torch.nn as nn

from .decoder_parts import OutConv, Up

class ProbabilityDecoder(nn.Module):
    def __init__(self, n_channels=288, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.initial_conv = nn.Conv2d(n_channels, 144, kernel_size=1)

        self.up1 = Up(144 + 512, 256, bilinear)
        self.up2 = Up(256 + 512, 128, bilinear)
        self.up3 = Up(128 + 256, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x, skips):

        skips = skips[::-1]

        x = self.initial_conv(x)

        x = self.up1(x, skips[0])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[2])

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.initial_conv = torch.utils.checkpoint(self.initial_conv)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
