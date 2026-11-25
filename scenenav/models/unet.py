"""Собственный U-Net с настраиваемым числом входных каналов (base_channels=32 — облегчённый вариант)."""
import torch
import torch.nn as nn


def _double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.conv = _double_conv(cin, cout)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, dropout: float = 0.0):
        super().__init__()
        c = base_channels
        self.in_channels = in_channels
        self.inc = _double_conv(in_channels, c)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), _double_conv(c, c * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), _double_conv(c * 2, c * 4))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), _double_conv(c * 4, c * 8))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), _double_conv(c * 8, c * 16), nn.Dropout2d(dropout))
        self.u1, self.u2 = _Up(c * 16, c * 8), _Up(c * 8, c * 4)
        self.u3, self.u4 = _Up(c * 4, c * 2), _Up(c * 2, c)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)
