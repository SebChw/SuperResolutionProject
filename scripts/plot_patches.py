import yaml
import argparse
import cv2
from pathlib import Path
import numpy as np


def plot_patches_randomly(config : dict):
    # cv2.namedWindow('patches', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('patches', 400,500)

    hr_folder = Path(config['data_root']) / Path(f"{config['prefix']}HR{config['new_suffix']}{config['patch_size']}")
    num_hr_patches = 50
    img = "0001"
    while True:
        random_patch = np.random.choice(num_hr_patches, 1)[0]
        target_path = hr_folder / f"{img}_{random_patch}.png"

        hr = cv2.imread(str(target_path))
        downscaling = np.random.choice(config['downscaling_types'],1)[0]
        images = [hr]
        for scale_factor in config['scaling_factors']:
            lr_folder = Path(config['data_root']) / Path(f"{config['prefix']}LR_{downscaling}/X{scale_factor}{config['new_suffix']}{config['patch_size']}")
            lr_path = lr_folder / f"{img}_{random_patch}x{scale_factor}.png"
            lr_image = cv2.imread(str(lr_path))
            background = np.zeros((hr.shape[0], lr_image.shape[1], 3), dtype=np.uint8)
            background[:lr_image.shape[0]] = lr_image
            
            images.append(background)

        cv2.imshow("patches", np.concatenate(images, axis=1))
        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'PatchesPlotter',
                    description = 'This small scripts will plot you randomly selected patches from different resolutions to compare them',
                    )
    parser.add_argument('-c','--config_path', default='configs/patches.yaml', required=False)
    args = parser.parse_args()

    with open(args.config_path, "r") as config:
        plot_patches_randomly(yaml.safe_load(config))
