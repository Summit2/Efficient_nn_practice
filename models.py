import numpy as np
import torch.nn as nn
import torch
class SmallCNN(nn.Module):
    """
    Small sequential CNN for Homework 1.

    Architecture (all convs: padding = k // 2, bias = False; ReLU inplace):
        Conv7x7 s2  3   -> 32   , MaxPool 3x3 s2 p1   -> S/4
        Conv5x5     32  -> 64                          -> S/4
        Conv3x3 s2  64  -> 128                         -> S/8
        Conv1x1     128 -> 256                         -> S/8
        Conv3x3 s2  256 -> 256                         -> S/16
        Conv1x1     256 -> 512                         -> S/16
        head: GlobalAvgPool -> Linear 512->256 -> ReLU -> Linear 256->100
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.features = nn.Sequential(
            # S -> S/2 (conv s2) -> S/4 (pool s2 p1)
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # S/4 -> S/4
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),

            # S/4 -> S/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # S/8 -> S/8
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            # S/8 -> S/16
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # S/16 -> S/16
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # GlobalAvgPool -> (B, 512, 1, 1)
            nn.Flatten(),              # (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

