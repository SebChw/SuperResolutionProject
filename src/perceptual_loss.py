from torchvision.models import vgg16, VGG16_Weights
import torch
from torchvision import transforms
import torch.nn as nn

from utils import minMaxTensor
from torch.nn import functional as F


class PerceptualLoss(nn.Module):
    def __init__(self,):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])  # values from PyTorch docs
        ])

        self.extractor = vgg16(
            weights=VGG16_Weights.IMAGENET1K_FEATURES).features

        i = 0

        self.model = nn.Sequential()

        use_maxpool = True

        end_layer = 16

        for layer in self.extractor:

            if i > end_layer:
                break

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name, layer)

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                if use_maxpool:
                    self.model.add_module(name, layer)
                else:
                    avgpool = nn.AvgPool2d(
                        kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                    self.model.add_module(name, avgpool)
            i += 1

        self.model = self.model.to("cuda")

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, sr_x, y):
        sr_x_features = self.model(self.transform(minMaxTensor(sr_x)))
        y_features = self.model(self.transform(minMaxTensor(y)))

        return F.mse_loss(sr_x_features, y_features)
