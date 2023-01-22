import torch.nn as nn
import torch

"""
    Implementation from paper
    https://arxiv.org/abs/1609.04802
    but discriminator is much smaller than in their paper
"""


def get_conv_block(in_channels, out_channels, stride, batch_norm):
    bias = False if batch_norm else True
    conv = nn.Conv2d(in_channels, out_channels, 3,
                     stride, padding=1, bias=bias)
    act = nn.LeakyReLU()

    if not batch_norm:
        return nn.Sequential(conv, act)
    else:
        return nn.Sequential(conv, nn.BatchNorm2d(out_channels), act)


class SRDiscriminator(nn.Module):
    def __init__(self,):
        super().__init__()

        # Input is 192x192 image

        self.first_conv = get_conv_block(3, 16, stride=1, batch_norm=False)

        # discriminator is half of the original SRResnet one
        conv_blocks_def = [
            [16, 32, 2],
            [32, 64, 2],
            [64, 128, 2],
        ]

        self.conv_block = nn.ModuleList([
            get_conv_block(in_c, out_c, stride=stride, batch_norm=True) for in_c, out_c, stride in conv_blocks_def
        ])

        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, X):
        X = self.first_conv(X)

        for c in self.conv_block:
            X = c(X)

        return self.head(X)


if __name__ == "__main__":
    d = SRDiscriminator()

    out = d(torch.randn((1, 3, 192, 192)))

    print(out.shape)
