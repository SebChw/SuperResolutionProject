
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
from torchvision import transforms
import torch.nn as nn

from torch.nn import functional as F

"""
    Perceptual loss from this paper https://arxiv.org/abs/1603.08155
    However, instead of vgg we used Efficient Net
"""


class PerceptualLoss(nn.Module):
    def __init__(self,):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])  # values from PyTorch docs
        ])

        self.extractor = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features

        i = 0

        self.model = nn.Sequential()

        end_layer = 3

        for layer in self.extractor:
            if i > end_layer:
                break

            self.model.add_module(f"layer{i}", layer)

            i += 1

        self.model = self.model.to("cuda")

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, sr_x, y):
        sr_x_features = self.model(self.transform(sr_x))
        y_features = self.model(self.transform(y))

        loss = F.mse_loss(sr_x_features, y_features)

        if torch.isnan(loss):
            print("NAN!")

        return F.mse_loss(sr_x_features, y_features)
