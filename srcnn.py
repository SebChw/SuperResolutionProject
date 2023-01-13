from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sr_dataset import SRDataset

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
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		sr_x = self.model(x)    
		loss = F.mse_loss(sr_x, x)
		self.log('val_loss', loss)

trainset = SRDataset(return_scaling_factor=False, perform_bicubic=True)
validset = SRDataset(train=False, return_scaling_factor=False, perform_bicubic=True)


train_loader = DataLoader(trainset, batch_size=32)
val_loader = DataLoader(validset, batch_size=32)

# model
model = LitSRCNN()

# training
trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16)
trainer.fit(model, train_loader, val_loader)
    
