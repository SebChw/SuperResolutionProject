import numpy as np
import cv2
from utils import collect_paths
from pathlib import Path
import shutil
PATCH_SIZE = 48
STEP_SIZE = 22

def save_patch(img, box, path):
    x1,y1,x2,y2 = box
    cv2.imwrite(path, img[y1:y2, x1:x2])
    

data_df = collect_paths("data", "DIV2K_train_", ["unknown"], [2,3,4], merge_inputs=True)


for id, row in data_df.iterrows():
    input_paths, target_path = row
    target_path = Path(target_path)
    target_img = cv2.imread(str(target_path))

    all_boxes = np.array([0,0,PATCH_SIZE,PATCH_SIZE]) + (np.array([[STEP_SIZE,0,STEP_SIZE,0]]) * np.expand_dims(np.arange(target_img.shape[0] // STEP_SIZE - 1),1))
    all_boxes = all_boxes[None, ...] + (np.array([[0,STEP_SIZE,0,STEP_SIZE]]) * np.expand_dims(np.arange(target_img.shape[0] // STEP_SIZE - 1),1))[:,None,:]
    all_boxes = all_boxes.reshape(-1,4)

    destination = Path(str(target_path.parent) + "_patches")
    shutil.rmtree(destination)
    destination.mkdir()
    for i, box in enumerate(all_boxes):
        name = f"{target_path.stem}_{i}.png"
        save_patch(target_img, box, str(destination / name))

    for input_path in input_paths:
        input_path = Path(input_path)
        input_img = cv2.imread(str(input_path))

        destination = Path(str(input_path.parent) + "_patches")
        shutil.rmtree(destination)
        scaling_factor = int(input_path.parts[-2][1])
        all_boxes_scaled = all_boxes // scaling_factor
        for i, box in enumerate(all_boxes_scaled):
            id_ ,scale = input_path.stem.split("x")
            name = f"{id_}_{i}x{scale}.png"
            save_patch(input_img, box, str(destination / name))
    
    break # when it is ready this should be taken off.


#print(collect_paths("data", "DIV2K_train_", ["unknown"], [2,3,4], merge_inputs=True, patches="_patches"))