import sys
import argparse
import yaml

from srcnn import LitSRCNN
from sr_dataset import SRDataset

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from custom_callbacks import ImageLoggingCallback


class Trainer:
    def __init__(self):
        self.architectures = {'srcnn': LitSRCNN}

    def run(self, config):
        self.parse_args(config)
        self.neptune_logger = NeptuneLogger(
            project="skdbmk/superresolution",
            tags=["training", self.architecture.__name__],  # optional
        )
        train_loader, val_loader = self.get_data(config)
        model = self.architecture(**config['model_parameters'])

        overfit_trainer = pl.Trainer(
            accelerator="gpu", devices=1, precision=16, overfit_batches=1, max_epochs=100)
        overfit_trainer.fit(model, train_loader, train_loader)

        trainer = pl.Trainer(accelerator="gpu", devices=1,
                             precision=16, logger=self.neptune_logger, callbacks=[
                                 ImageLoggingCallback()],
                             val_check_interval=0.25, max_epochs=5)

        trainer.logger.experiment['config'] = config

        trainer.fit(model, train_loader, val_loader)
        # input_sample = torch.randn((192,192,3))
        # model.to_onnx(
        #     f"model_{self.architecture.__name__}.onnx", input_sample, export_params=True)
        # TODO take a few random images from traindataset and create before -> after

    def parse_args(self, config):
        self.scaling_factors = config["scaling_factors"]
        architecture_type = config["architecture"]
        self.batch_size = config["batch_size"]
        if architecture_type in self.architectures.keys():
            self.architecture = self.architectures[architecture_type]
        else:
            print('Architecture not implemented')
            sys.exit()

    def get_data(self, config):
        trainset = SRDataset(perform_bicubic=config['train_pre_sampling'],
                             patches=config['patches'],
                             extension=config['train_extension'],
                             random_dataset_order=config['random_dataset_order'],
                             scaling_factors=self.scaling_factors)
        validset = SRDataset(
            train=False,
            perform_bicubic=config['val_pre_sampling'],
            extension=config['val_extension'],
            random_dataset_order=config['random_dataset_order'],
            scaling_factors=self.scaling_factors)

        #! Don't set shuffle to True
        train_loader = DataLoader(
            trainset, batch_size=32, num_workers=0)
        # ! On windows every worker loads everything to RAM so it is hard to have many withouth tricks

        val_loader = DataLoader(validset, batch_size=1, num_workers=0)
        return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/train_srcnn.yaml', required=False)
    args = parser.parse_args()
    Trainer = Trainer()
    with open(args.config_path, "r") as config:
        Trainer.run(yaml.safe_load(config))
