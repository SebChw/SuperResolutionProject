import sys
import argparse
import yaml

from srcnn import LitSRCNN
from sr_dataset import SRDataset
from pl_bolts.models.gans.srgan.srresnet_module import SRResNet
from pl_bolts.models.gans.srgan.srgan_module import SRGAN

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from custom_callbacks import ImageLoggingCallback


# liniowa interpolacja nie ma sensu
# Zrobic ewaluacje przed sama siecia.
# UCZYC TYLKO X2
# Nawet dla presamplingu tylko X2

class Trainer:
    def __init__(self):
        self.architectures = {'srcnn': LitSRCNN,
                              'srresnet': SRResNet,
                              'srgan': SRGAN}

    def run(self, config):
        self.parse_args(config)
        self.neptune_logger = NeptuneLogger(
            project="skdbmk/superresolution",
            tags=["training", self.architecture.__name__],  # optional
        )
        train_loader, val_loader = self.get_data(config)
        self.model = self.architecture(**config['model_parameters'])

        # self.overfit_one_batch(config)

        trainer = pl.Trainer(accelerator="gpu", devices=1,
                             precision=16, logger=self.neptune_logger, callbacks=[
                                 ImageLoggingCallback(config)], max_epochs=5)

        trainer.logger.experiment['config'] = config

        trainer.fit(self.model, train_loader, val_loader)
        # input_sample = torch.randn((192,192,3))
        # model.to_onnx(
        #     f"model_{self.architecture.__name__}.onnx", input_sample, export_params=True)
        # TODO take a few random images from traindataset and create before -> after

    def parse_args(self, config):
        architecture_type = config["architecture"]
        if architecture_type in self.architectures.keys():
            self.architecture = self.architectures[architecture_type]
        else:
            print('Architecture not implemented')
            sys.exit()

    def overfit_one_batch(self, config):
        trainset = SRDataset(
            **config['dataset_parameters'], patches=config['patches'])
        train_loader = DataLoader(
            trainset, batch_size=32, num_workers=0)

        overfit_trainer = pl.Trainer(
            accelerator="gpu", devices=1, precision=16, overfit_batches=1, max_epochs=100)
        overfit_trainer.fit(self.model, train_loader, train_loader)

    def get_data(self, config):
        trainset = SRDataset(
            patches=config['patches'], **config['dataset_parameters'])
        validset = SRDataset(
            train=False, **config['dataset_parameters'])

        #! If you train it on HDD don't set shuffle to true
        train_loader = DataLoader(
            trainset, batch_size=config["batch_size"], shuffle=config['shuffle'], num_workers=config['num_workers_train'])
        # ! On windows every worker loads everything to RAM so it is hard to have many withouth tricks

        val_loader = DataLoader(validset, batch_size=1,
                                num_workers=config['num_workers_valid'])
        return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/train_srresnet.yaml', required=False)
    args = parser.parse_args()
    Trainer = Trainer()
    with open(args.config_path, "r") as config:
        Trainer.run(yaml.safe_load(config))
