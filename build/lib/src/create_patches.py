import numpy as np
import cv2
from torchsr.utils import collect_paths
from pathlib import Path
import argparse
import yaml


def save_patch(img, box, path, save_function):
    #! For future this would be much, much better if patches were saved as a binary files!
    x1, y1, x2, y2 = box

    if y2 <= img.shape[0] and x2 <= img.shape[1]:
        save_function(path, img[y1:y2, x1:x2])
    else:
        offset_x = x2 - img.shape[1] if x2 > img.shape[1] else 0
        offset_y = y2 - img.shape[0] if y2 > img.shape[0] else 0
        save_function(path, img[y1-offset_y:y2 -
                      offset_y, x1-offset_x:x2-offset_x])


def save_numpy(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.moveaxis(img, -1, 0)
    np.save(path, img)


def cut_patches(config):
    PATCH_SIZE = config['patch_size']
    STEP_SIZE = config['step_size']
    DATA_ROOT = config['data_root']
    PREFIX = config['prefix']
    DOWNSAMPLIN_TYPES = config['downscaling_types']
    SCALING_FACTORS = config['scaling_factors']
    NEW_SUFFIX = f"{config['new_suffix']}{PATCH_SIZE}"
    EXTENSION = config['extension']

    saving_function = save_numpy if EXTENSION == "npy" else cv2.imwrite
    # this collect all files from given folder, so its validation
    # for unknown downscaling and for all downscaling factors, additionally it merges inputs so instead 3 rows we have 1 it looks like that
    # input_paths, target_path
    # [im2x2, im2x3, im2x4], im2
    data_df = collect_paths(
        DATA_ROOT, PREFIX, DOWNSAMPLIN_TYPES, SCALING_FACTORS, merge_inputs=True)

    # Create folders if needed
    for s_factor in SCALING_FACTORS:
        for d_type in DOWNSAMPLIN_TYPES:
            (Path(DATA_ROOT) / Path(f"{PREFIX}LR_{d_type}") /
             Path(f"X{s_factor}{NEW_SUFFIX}")).mkdir(exist_ok=True)

    (Path(DATA_ROOT) / Path(f"{PREFIX}HR{NEW_SUFFIX}")).mkdir(exist_ok=True)

    for id, row in data_df.iterrows():
        input_paths, target_path = row

        target_img = cv2.imread(str(target_path))
        print(target_path)
        target_path = Path(target_path)

        # Here we automatically create all possible bounding boxes for target image
        #!with this CEIL there will be one window that actually may be too big! and go over the image
        # this must be handled later by creating an offset
        all_boxes = np.array([0, 0, PATCH_SIZE, PATCH_SIZE]) +\
            (np.array([[STEP_SIZE, 0, STEP_SIZE, 0]]) * np.expand_dims(
                np.arange(int(np.ceil(target_img.shape[1] / STEP_SIZE))), 1))
        all_boxes = all_boxes[None, ...] +\
            (np.array([[0, STEP_SIZE, 0, STEP_SIZE]]) * np.expand_dims(
                np.arange(int(np.ceil(target_img.shape[0] / STEP_SIZE))), 1))[:, None, :]
        all_boxes = all_boxes.reshape(-1, 4)

        assert all_boxes.shape[0] == np.ceil(
            target_img.shape[0] / STEP_SIZE) * np.ceil(target_img.shape[1] / STEP_SIZE)
        # We will save them in a folder with same name but _patches suffix
        destination = Path(str(target_path.parent) + NEW_SUFFIX)

        for i, box in enumerate(all_boxes):
            name = f"{target_path.stem}_{i}.{EXTENSION}"
            save_patch(target_img, box, str(
                destination / name), saving_function)

        for input_path in input_paths:
            input_img = cv2.imread(str(input_path))
            input_path = Path(input_path)

            destination = Path(str(input_path.parent) + NEW_SUFFIX)
            scaling_factor = int(input_path.parts[-2][1])
            # All boxes dimensions are scaled down by appropriate scaling_factor
            if config['perform_bicubic_interpolation']:
                all_boxes_scaled = all_boxes.copy()
                input_img = cv2.resize(
                    input_img, target_img.shape[-2::-1], interpolation=cv2.INTER_CUBIC)
            else:
                all_boxes_scaled = all_boxes // scaling_factor

            for i, box in enumerate(all_boxes_scaled):
                id_, scale = input_path.stem.split("x")
                # we save them in a form that scale is at the very end, so this is consistent.
                name = f"{id_}_{i}x{scale}.{EXTENSION}"
                save_patch(input_img, box, str(
                    destination / name), saving_function)
        if id > 10:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PatchesCutter',
        description='Iterates over all HR images and corresponding downsampled ones and create patches of size given in config',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/patches.yaml', required=False)
    args = parser.parse_args()

    with open(args.config_path, "r") as config:
        cut_patches(yaml.safe_load(config))
