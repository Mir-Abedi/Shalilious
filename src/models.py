"""The networks. Every one takes (in_ch, n_classes, size), so one registry serves both
datasets and nothing else in the code branches on which is loaded."""
import torch.nn as nn
import torchvision


def small_cnn(in_ch=3, n_classes=100, size=32):
    """Two conv-pool blocks and two FC layers. Fast enough to debug on a laptop.

    At (1, 10, 28) this is the standard MNIST CNN -- two 3x3 blocks into a 256-unit head.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * (size // 4) ** 2, 256), nn.ReLU(),
        nn.Linear(256, n_classes),
    )


def linear(in_ch=1, n_classes=10, size=28):
    """Multinomial logistic regression: no hidden layer, so the objective is CONVEX.

    Worth having as an arm: Local SGD's convergence rates are stated for convex
    objectives, so here the measured K- and heterogeneity-penalties can be checked
    against a prediction rather than only compared across settings.
    """
    return nn.Sequential(nn.Flatten(), nn.Linear(in_ch * size * size, n_classes))


def resnet18(in_ch=3, n_classes=100, size=None):
    """torchvision ResNet-18 with the standard small-image adaptation: 3x3 stem, no maxpool.

    `size` is ignored -- the adaptive average pool before the head takes any input size.
    """
    m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m
