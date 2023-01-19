from pathlib import Path
import pandas as pd
import torch
import numpy as np


def collect_paths(data_path, prefix, downscalings=["unknown"], scaling_factors=[2, 3, 4], merge_inputs=False, patches="", extension="png"):
    """This function iterates over all HR images and find corresponding LR images at the end it returns data frame
       Where this paths are connected

    Args:
        data_path (str): path to main folder with data
        prefix (str): prefix of the data folder like, e.g DIV2K_train if we are interested in training data
        downscalings (list): which downscaling to gather, e.g ["unknown", "bicubic"] It's just used for a path creation
        scaling_factors (list): list with scaling factors e.g [2,4] used for path creations 
        merge_inputs (bool, optional): whether to create n separate rows or one row with list of n elements as a entry. Defaults to False.
        patches (str, optional): additional suffix to data folder. Defaults to "".

    Returns:
        pd.DataFrame
    """
    #! THIS patches variable is just to add _patches prefix if needed.
    targets = []
    inputs = []
    scalings = []
    for downscaling in downscalings:
        x_folder = Path(data_path) / Path(f"{prefix}LR_{downscaling}")
        y_folder = Path(data_path) / Path(f"{prefix}HR{patches}")

        for file in y_folder.iterdir():
            img_id = file.stem
            for scaling_factor in scaling_factors:
                inputs.append(
                    x_folder / f"X{scaling_factor}{patches}" / f"{img_id}x{scaling_factor}.{extension}")
                targets.append(file)
                scalings.append(scaling_factor)

        if not merge_inputs:
            return pd.DataFrame(list(zip(inputs, targets, scalings)), columns=["input_path", "target_path", "scaling_factor"])

        break  # ! since I have just 1 example available
    inputs = [inputs[i:i+3] for i in range(0, len(inputs), 3)]
    targets = [targets[i] for i in range(0, len(targets), 3)]
    return pd.DataFrame(list(zip(inputs, targets)), columns=["input_paths", "target_path"])


def cut_tensor_from_0_to_1(tensor: torch.tensor):
    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1

    return tensor
