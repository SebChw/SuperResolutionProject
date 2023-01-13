from pathlib import Path
import pandas as pd

def collect_paths(data_path, prefix, downscalings, scaling_factors, merge_inputs=False, patches = ""):
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
                inputs.append(x_folder / f"X{scaling_factor}{patches}" / f"{img_id}x{scaling_factor}.png")
                targets.append(file)
                scalings.append(scaling_factor)

        if not merge_inputs:
            return pd.DataFrame(list(zip(inputs, targets, scalings)), columns=["input_path", "target_path", "scaling_factor"])

        break #! since I have just 1 example available
    inputs = [inputs[i:i+3] for i in range(0,len(inputs),3)]
    targets = [targets[i] for i in range(0,len(targets),3)]
    return pd.DataFrame(list(zip(inputs, targets)), columns = ["input_paths", "target_path"])