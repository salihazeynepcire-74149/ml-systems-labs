"""CNN extracted verbatim from the supplied Lab 3 notebook."""
import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)

Lab3CNN = CNN

def build_model(cfg):
    if cfg['model']['architecture'] != 'lab3_cnn':
        raise ValueError('Only lab3_cnn is available; replace this factory with your best Lab 3 model.')
    return CNN()


def model_count(model):
    return sum(p.numel() for p in model.parameters())
