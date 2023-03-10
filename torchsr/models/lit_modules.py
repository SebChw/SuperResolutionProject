import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn as nn

from torchsr.models.discriminator import SRDiscriminator
from torchsr.utils.perceptual_loss import PerceptualLoss
from torchsr.utils.utils import cut_tensor_from_0_to_1
from torchsr.data.data_augmentation import DataAugmentation


class LitGenerator(pl.LightningModule):
    """ Class implementing pure generator - module that takes low resultion image and produces high resolution
    """

    def __init__(self, generator, generator_parameters):
        super().__init__()
        self.save_hyperparameters(ignore=['generator'])
        self.augment_train = generator_parameters["augment_train"]
        self.loss = generator_parameters["loss"]

        self.generator = generator

        self.valid_psnr = PeakSignalNoiseRatio()
        self.valid_ssim = StructuralSimilarityIndexMeasure()

        self.augmentations = DataAugmentation()

        if self.loss == "L2":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss

        if generator_parameters["perceptual_loss"]:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(self, x):
        sr_X = self.generator(x)
        return sr_X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        return optimizer

    def get_training_loss(self, sr_x, y):
        loss = self.loss(sr_x, y)
        self.log('train_loss', loss, prog_bar=True)

        loss_perc = 0
        if self.perceptual_loss:
            loss_perc = self.perceptual_loss(sr_x, y)
            self.log("train_perceptual_loss", loss_perc)

        total_loss = loss + loss_perc*0.005

        self.log("total_loss", total_loss)

        return total_loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.augment_train:
            x, y = self.augmentations(x, y)
        sr_x = self.generator(x)

        return self.get_training_loss(sr_x, y)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        sr_x = self.generator(x)
        loss = self.loss(sr_x, y)
        self.log('val_loss', loss)

        self.valid_psnr(sr_x, y)
        self.log('valid_psnr', self.valid_psnr, on_step=True, on_epoch=True)

        # ssim seems not to work when y is float 16
        # and also as we move beyond 0 and 1
        # But still there was something wrong and some images gave weird results
        self.valid_ssim(cut_tensor_from_0_to_1(sr_x).to(torch.float32), y)
        self.log('valid_ssim', self.valid_ssim, on_step=True, on_epoch=True)


class LitSRGan(LitGenerator):
    """
        Extdens pure generator with discriminator and adversarial_loss so it uses GAN strategy to learn SR
    """

    def __init__(self, generator, generator_parameters):
        super().__init__(generator, generator_parameters)

        self.discriminator = SRDiscriminator()
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        x, y = train_batch

        if self.augment_train:
            x, y = self.augmentations(x, y)

        # generate images
        sr_x = self.generator(x)

        # train generator
        if optimizer_idx == 0:
            discriminator_d = self.discriminator(sr_x)

            valid = torch.ones_like(discriminator_d)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(discriminator_d, valid)
            self.log("g_loss", g_loss, prog_bar=True, sync_dist=True)

            reconstruction_loss = self.get_training_loss(sr_x, y)

            return g_loss + reconstruction_loss

        # train discriminator
        if optimizer_idx == 1:
            discrminator_real = self.discriminator(y)

            # how well can it label as real?
            valid = torch.ones_like(discrminator_real)

            real_loss = self.adversarial_loss(discrminator_real, valid)

            discriminator_fake = self.discriminator(sr_x.detach())

            fake = torch.zeros_like(discriminator_fake)

            fake_loss = self.adversarial_loss(discriminator_fake, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=1e-3)
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-3)  # lr 1e-5 is too small and nothing happens

        return [opt_g, opt_d], []
