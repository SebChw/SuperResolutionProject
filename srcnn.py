from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils import cut_tensor_from_0_to_1
from data_augmentation import DataAugmentation
"""
Here we have very simple SR model and very basic Lightning pipeline
"""


class SRCNN(nn.Module):
    def __init__(self, kernel_sizes=[9, 1, 5], num_filters=[64, 32]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.model = nn.Sequential(
            nn.Conv2d(
                3, num_filters[0], kernel_size=kernel_sizes[0], padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1],
                      kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], 3,
                      kernel_size=kernel_sizes[2], padding="same")
        )

    def forward(self, X):
        #! At this moment we assume that images have been put through bicubic interpolation
        #! and has the same size as expected result.
        return self.model(X)


class LitSRCNN(pl.LightningModule):
    def __init__(self, augment_train=True, loss="L2"):
        super().__init__()
        self.model = SRCNN()

        self.train_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()

        self.valid_psnr = PeakSignalNoiseRatio()
        self.valid_ssim = StructuralSimilarityIndexMeasure()

        self.augmentations = DataAugmentation()
        self.augment_train = augment_train

        if loss == "L2":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss

    def forward(self, x):
        # This should be something like inference step, according to lightning documentation
        sr_X = self.model(x)
        return sr_X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.augment_train:
            x, y = self.augmentations(x, y)
        sr_x = self.model(x)

        loss = self.loss(sr_x, y)
        self.log('train_loss', loss)

        #! not sure if we want this during training
        # self.train_psnr(sr_x, y)
        # self.log('train_psnr', self.train_psnr, on_step=True, on_epoch=False)
        # self.train_ssim(sr_x.to(torch.float32), y)
        # self.log('train_ssim', self.train_psnr, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        sr_x = self.model(x)
        loss = F.mse_loss(sr_x, x)
        self.log('val_loss', loss)

        self.valid_psnr(sr_x, y)
        self.log('valid_psnr', self.valid_psnr, on_step=True, on_epoch=True)
        #! ssim seems to work quite bad

        # ssim seems not to work when y is float 16
        self.valid_ssim(cut_tensor_from_0_to_1(sr_x).to(torch.float32), y)
        self.log('valid_ssim', self.valid_ssim, on_step=True, on_epoch=True)
