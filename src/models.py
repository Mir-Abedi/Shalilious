"""The two networks. Both return 100 logits; nothing else in the code branches on which."""
import torch.nn as nn
import torchvision


def small_cnn():
    """Two conv-pool blocks and two FC layers. Fast enough to debug on a laptop."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
        nn.Linear(256, 100),
    )


def resnet18():
    """torchvision ResNet-18 with the standard 32x32 adaptation: 3x3 stem, no maxpool."""
    m = torchvision.models.resnet18(weights=None, num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m
