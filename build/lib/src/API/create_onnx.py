from src.srcnn import get_srcnn
import argparse
import yaml
import torch


def create_onnx(config):
    model = get_srcnn(config['model_parameters'])
    model = model.load_from_checkpoint(
        f"trained_models/{config['architecture']}.ckpt", strict=False)
    model.eval()
    # model.pred
    # input_sample = torch.randn((3, 192, 192))
    # model.to_onnx(f"trained_models/{config['architecture']}.onnx", input_sample, export_params=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/train_srcnn.yaml', required=False)
    args = parser.parse_args()
    with open(args.config_path, "r") as config:
        create_onnx(yaml.safe_load(config))
