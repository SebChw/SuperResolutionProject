import numpy as np
import cv2
from utils import collect_paths
from pathlib import Path
import shutil

"""This script goes through HR images cut square of PATCH_SIZE and then iterate over all corresponding
   LR images and cut appropriate square so that there is correspondence between HR and LR image.

   This is done as different images have different shapes, to put all in a batch.

   PATCH_SIZE SHOULD BE NO SMALLER THAN RECEPTIVE FIELD OF A MODEL! 
"""
PATCH_SIZE = 48
STEP_SIZE = 22

def save_patch(img, box, path):
    x1,y1,x2,y2 = box
    cv2.imwrite(path, img[y1:y2, x1:x2])
    
#this collect all files from given folder, so its validation
#for unknown downscaling and for all downscaling factors, additionally it merges inputs so instead 3 rows we have 1 it looks like that
#input_paths, target_path
#[im2x2, im2x3, im2x4], im2 
data_df = collect_paths("data", "DIV2K_valid_", ["unknown"], [2,3,4], merge_inputs=True)


for id, row in data_df.iterrows():
    input_paths, target_path = row
    target_img = cv2.imread(target_path)

    target_path = Path(target_path)

    # Here we automatically create all possible bounding boxes for target image
    #TODO test if possible
    all_boxes = np.array([0,0,PATCH_SIZE,PATCH_SIZE]) + (np.array([[STEP_SIZE,0,STEP_SIZE,0]]) * np.expand_dims(np.arange(target_img.shape[0] // STEP_SIZE - 1),1))
    all_boxes = all_boxes[None, ...] + (np.array([[0,STEP_SIZE,0,STEP_SIZE]]) * np.expand_dims(np.arange(target_img.shape[0] // STEP_SIZE - 1),1))[:,None,:]
    all_boxes = all_boxes.reshape(-1,4)

    #We will save them in a folder with same name but _patches suffix
    destination = Path(str(target_path.parent) + "_patches")

    for i, box in enumerate(all_boxes):
        name = f"{target_path.stem}_{i}.png"
        save_patch(target_img, box, str(destination / name))

    for input_path in input_paths:
        input_img = cv2.imread(input_path)
        input_path = Path(input_path)
        
        destination = Path(str(input_path.parent) + "_patches")
        scaling_factor = int(input_path.parts[-2][1])
        #All boxes dimensions are scaled down by appropriate scaling_factor
        all_boxes_scaled = all_boxes // scaling_factor
        for i, box in enumerate(all_boxes_scaled):
            id_ ,scale = input_path.stem.split("x")
            name = f"{id_}_{i}x{scale}.png" # we save them in a form that scale is at the very end, so this is consistent.
            save_patch(input_img, box, str(destination / name))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    break # when it is ready this should be taken off.
