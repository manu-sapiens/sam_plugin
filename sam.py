# --------------- SHARED --------------------------------------------
print(f"RUNNING: {__name__} ")
import sys
sys.path.append('.')  # Add the current directory to the sys path
sys.path.append('utils')  # Add the current directory to the sys path
from utils.omni_utils_misc import omni_get_env
from utils.omni_utils_http import CdnHandler, CdnResponse, ImageMeta, route_commands
routes_info = {}
# ------------------------------------------------------------------
OMNI_TEMP_FOLDER = omni_get_env("OMNI_TEMP_FOLDER")
OMNI_CHECKPOINT_FOLDER = omni_get_env("OMNI_CHECKPOINT_FOLDER")

from utils.omni_utils_gpu import GetTorchDevice

import os
from utils.omni_utils_http import CdnHandler
from utils.omni_utils_masks import process_mask, save_image_or_mask
from typing import Any, List
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from fastapi import HTTPException

# ---------------------------------------------------
# --------------- SAM GENERATE MASKS ----------------
# ---------------------------------------------------
SAM_MODEL_CHECKPOINT_VIT_H = "sam_vit_h_4b8939.pth" # large
SAM_MODEL_CHECKPOINT_VIT_L = "sam_vit_l_0b3195.pth" # medium
SAM_MODEL_CHECKPOINT_VIT_B = "sam_vit_b_01ec64.pth" # small
SAM_MODEL_VIT_H = "vit_h"
SAM_MODEL_VIT_L = "vit_l"
SAM_MODEL_VIT_B = "vit_b"
sams = {}

def SamGetModelCheckpoint_Init(model_type:str):

    checkpoint_path = SAM_MODEL_CHECKPOINT_VIT_L
    model_checkpoints = {
        SAM_MODEL_VIT_H: SAM_MODEL_CHECKPOINT_VIT_H,
        SAM_MODEL_VIT_L: SAM_MODEL_CHECKPOINT_VIT_L,
        SAM_MODEL_VIT_B: SAM_MODEL_CHECKPOINT_VIT_B
    }

    if model_type in model_checkpoints:
        print("model_type found.")
        checkpoint_path = os.path.join(OMNI_CHECKPOINT_FOLDER, model_checkpoints[model_type])
    else:
        print("[WARNING] model_type not found. Using default.")

    #
    print(f"checkpoint_path = {checkpoint_path}")

    return checkpoint_path

def SamGetModel_Init(model_type:str):
    if model_type not in sams:        
        print(f"{model_type} not loaded yet. Loading...")
        checkpoint = SamGetModelCheckpoint_Init(model_type)
        device = GetTorchDevice()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        print("Loaded.")

        sams[model_type] = sam
    
    return sams[model_type]

async def generate_masks_or_images(cdn: CdnHandler, input_filename: str, model_type: str, output_mask: bool, output_merged: bool, output_alpha: bool, invert_mask: bool, minimum_score: float, detections: List[Any]):
    
    print("-------- action_sam_generate_masks ------")
    #print(f"detections = {detections}")
    print(f"model_type = {model_type}")
    print(f"output_merged = {output_merged}")
    print(f"output_alpha = {output_alpha}")
    print(f"invert_mask = {invert_mask}")
    print(f"minimum_score = {minimum_score}")
    #print(f"yolov_detections = {detections}")

    image_bgr = cv2.imread(input_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # SAM ----------------------------------------
    print("Initializing the SAM predictor")
    sam = SamGetModel_Init(model_type)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    result_detections = []
    result_filenames = []
    result_mask_filenames = []
    category_count = {}
    base_name ="sam_"+cdn.generate_random_name()
    max_x = image_bgr.shape[0]
    max_y = image_bgr.shape[1]
    merged_mask = None
    for detection_obj in detections:
        print(f"\nParsing detection_obj = {detection_obj}")

        detection = detection_obj["detection"]
        print(f"detection = {detection}")

        
        xyxy = detection["xyxy"]
        category = detection["category"]
  
        # Build input box in xyxy format
        input_box = np.array(xyxy)

        # Use input box as input for your predictor
        print(f"Using sam on box: {input_box}")
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )

        # Find the index of the mask with the highest score
        best_mask_index = np.argmax(scores)

        # Select the best mask and its corresponding score
        best_mask = masks[best_mask_index]
        best_score = scores[best_mask_index]
        print(f"best_score = {best_score}")

        if best_score >= minimum_score:
            filename, merged_mask = process_mask(best_mask, merged_mask, category, output_mask, output_merged, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name)
            print(f"identified: {filename}")
            result_filenames.append(filename)
       
            if False:
                print(f"output_merged = {output_merged}")
                print(type(output_merged))
                if output_merged:
                    if merged_mask is None:
                        merged_mask = best_mask.copy()
                    else:
                        merged_mask = np.logical_or(merged_mask, best_mask)  
                else:
                    result_detections.append(detection)
                    filename_image, filename_mask = SaveImageAndMask(best_mask, category, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name)
                    print(f"identified: {filename_image, filename_mask}")
                    result_filenames.append(filename_image)
                    result_mask_filenames.append(filename_mask)
                #
            #
    #

    if output_merged:

        print("Processing everthing mask")
        print(f"merged_mask.shape = {merged_mask.shape}")
        everything_filename = save_image_or_mask(merged_mask, "all", output_mask, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name+"_everything")
              
        result_filenames = [everything_filename]
        #result_mask_filenames = [everything_mask_filename]

        everything_detection = {"detection": {
                    'xyxy': [0, 0, max_x, max_y],
                    'confidence': 1.0,
                    'class_id': -1,
                    'category': "all"
                }
            }
    
        result_detections= [everything_detection]
    #


    return result_filenames, result_detections

def ProcessMask(mask, merged_mask, category, output_merged, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name):
    if output_merged:
        if merged_mask is None:
            merged_mask = mask.copy()
        else:
            merged_mask = np.logical_or(merged_mask, mask)  
    else:
        filename_image, filename_mask = SaveImageAndMask(mask, category, output_alpha, invert_mask, image_rgb, image_bgr, category_count, base_name)
        print(f"identified: {filename_image, filename_mask}")
    #

    return filename_image, filename_mask, merged_mask

def SaveImageAndMask(sam_mask: Any, category: str, output_alpha: bool, invert_mask: bool , image_rgb: Any, image_bgr:Any , category_count:int, base_name:str):
    output_image = None
    mask_image = None

    mask_image = np.zeros_like(image_rgb, dtype=np.uint8)
    mask_image[sam_mask] = [255, 255, 255]  # Set mask pixels to white

    if invert_mask: mask_image = np.bitwise_not(mask_image)  # Invert the mask image

    if category not in category_count: category_count[category] = 0
    i = category_count[category]

    filename_mask = os.path.join(OMNI_TEMP_FOLDER, base_name+"_"+f'mask_{category}_{i}.png')
    output_mask = mask_image
    filename_image = os.path.join(OMNI_TEMP_FOLDER, base_name+"_"+f'masked_{category}_{i}.png')
    output_image = cv2.bitwise_and(image_bgr, mask_image)

    if output_alpha:
        alpha_mask = mask_image[:, :, 0]  # Use the first channel of mask_image as alpha channel
        alpha_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        alpha_image[:, :, 3] = alpha_mask

        output_mask = np.concatenate((output_mask, alpha_mask[:, :, None]), axis=-1)  # Add the alpha channel
        output_image = alpha_image  # The alpha_image already has the alpha channel

    cv2.imwrite(filename_mask, output_mask)
    cv2.imwrite(filename_image, output_image)
    category_count[category] = category_count.get(category, 0) + 1

    return filename_image, filename_mask

from Plugins.sam_plugin.sam_plugin import SamGenerateMasks_Input, SamGenerateMasks_Response, ENDPOINT_SAM_GENERATE_MASKS
async def integration_SamGenerateMask_Post(input: SamGenerateMasks_Input):
    cdn = CdnHandler()
    if True: #try:
        cdn.announcement()
        print("------------- generate_masks ------------------")
        #print(f"input = {input}")

        detections = input.detections
        model_type = input.model_type
        minimum_score = input.minimum_score
        output_merged = input.output_merged 
        invert_mask = input.invert_mask
        output_alpha = input.output_alpha
        output_mask = input.output_mask
    

        # validation -------------------------------
        if model_type not in (SAM_MODEL_VIT_H, SAM_MODEL_VIT_L, SAM_MODEL_VIT_B): model_type = SAM_MODEL_VIT_L # default

        print(f"model_type = {model_type}")
        print(f"output_mask = {output_mask}")
        print(f"output_merged = {output_merged}")
        print(f"output_alpha = {output_alpha}")
        print(f"invert_mask = {invert_mask}")

        print(f"minimum_score = {minimum_score}")
        #print(f"yolov_detections = {detections}")

        input_cdns = [input.images[0]] # !!!!!!  
        input_filenames = await cdn.download_files_from_cdn(input_cdns)
        input_filename = input_filenames[0]
        # todo: we only support one image at a time for now as we already return an array of images and json for each processed image.
        # and I don't want to deal with arrays of array in the Designer (for now)

        # ------ sam
        sam_filenames, results_detections = await generate_masks_or_images(cdn=cdn, input_filename=input_filename, model_type=model_type, output_mask=output_mask, output_merged=output_merged, output_alpha=output_alpha, invert_mask=invert_mask, minimum_score=minimum_score, detections=detections)

        if len(sam_filenames) > 0:
            print(f"Uploading # {len(sam_filenames)}")
            results_cdns = await cdn.upload_files_to_cdn(sam_filenames)
            # delete the results files from the local storage
            cdn.delete_temp_files(sam_filenames)
        #

        response = SamGenerateMasks_Response(media_array=results_cdns, json_array=results_detections)
        print("\n-----------------------\n")
        return response
    else: #except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
routes_info[ENDPOINT_SAM_GENERATE_MASKS] = (SamGenerateMasks_Input, integration_SamGenerateMask_Post)

# --------------- SHARED -------------------------------------------
if __name__ == '__main__':
    route_commands(routes_info, sys.argv)
# ------------------------------------------------------------------