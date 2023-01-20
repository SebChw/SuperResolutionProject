import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils import cut_tensor_from_0_to_1, minMaxTensor
from data_augmentation import DataAugmentation


class LitGenerator(pl.LightningModule):
    def __init__(self, model, model_parameters):
        super().__init__()
        self.augment_train = model_parameters["augment_train"]
        self.loss = model_parameters["loss"]

        self.model = model

        self.train_psnr = PeakSignalNoiseRatio()
        self.train_ssim = StructuralSimilarityIndexMeasure()

        self.valid_psnr = PeakSignalNoiseRatio()
        self.valid_ssim = StructuralSimilarityIndexMeasure()

        self.augmentations = DataAugmentation()

        if self.loss == "L2":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss

    def forward(self, x):
        # This should be something like inference step, according to lightning documentation
        sr_X = self.model(x)
        return sr_X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.augment_train:
            x, y = self.augmentations(x, y)
        sr_x = self.model(x)

        loss = self.loss(sr_x, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        sr_x = self.model(x)
        loss = self.loss(sr_x, y)
        self.log('val_loss', loss)

        self.valid_psnr(sr_x, y)
        self.log('valid_psnr', self.valid_psnr, on_step=True, on_epoch=True)

        # ssim seems not to work when y is float 16
        # and also as we move beyond 0 and 1
        self.valid_ssim(minMaxTensor(sr_x).to(torch.float32), y)
        self.log('valid_ssim', self.valid_ssim, on_step=True, on_epoch=True)
