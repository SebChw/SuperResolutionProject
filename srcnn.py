from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sr_dataset import SRDataset
from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from pytorch_lightning.loggers import NeptuneLogger
"""
Here we have very simple SR model and very basic Lightning pipeline
"""
#! key returned by this i wrong I had to copy it from env variable to make it work!
from neptune.new import ANONYMOUS_API_TOKEN
neptune_logger = NeptuneLogger(
    api_key=ANONYMOUS_API_TOKEN,
	project="skdbmk/superresolution",
    tags=["training", "srcnn"],  # optional
)

class SRCNN(nn.Module):
    def __init__(self, kernel_sizes = [9,1,5], num_filters = [64, 32]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.model = nn.Sequential(
            nn.Conv2d(3, num_filters[0], kernel_size = kernel_sizes[0], padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size = kernel_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], 3, kernel_size=kernel_sizes[2], padding="same")
        )


    def forward(self, X):
        #! At this moment we assume that images have been put through bicubic interpolation
        #! and has the same size as expected result.
        return self.model(X)


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class LitSRCNN(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = SRCNN()

		self.train_psnr = PeakSignalNoiseRatio()
		self.train_ssim = StructuralSimilarityIndexMeasure()

		self.valid_psnr = PeakSignalNoiseRatio()
		self.valid_ssim = StructuralSimilarityIndexMeasure()

	def forward(self, x):
        #This should be something like inference step, according to lightning documentation
		sr_X = self.model(x)
		return sr_X

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		sr_x = self.model(x)    
		loss = F.mse_loss(sr_x, x)
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
		self.valid_ssim(sr_x.to(torch.float32), y) #ssim seems not to work when y is float 16
		self.log('valid_ssim', self.valid_ssim, on_step=True, on_epoch=True)

trainset = SRDataset(return_scaling_factor=False, perform_bicubic=True, patches="_patches")
validset = SRDataset(train=False, return_scaling_factor=False, perform_bicubic=True)


train_loader = DataLoader(trainset, batch_size=32)
val_loader = DataLoader(validset, batch_size=1) #! here we evaluate work on big images!

# model
model = LitSRCNN()

# training
trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, logger=neptune_logger)
trainer.fit(model, train_loader, val_loader)
    
