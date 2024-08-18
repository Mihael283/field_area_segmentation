import numpy as np
import os
import json
from shapely.geometry import Polygon
import pandas as pd
import itertools
import tifffile as tiff
from skimage import measure
from tqdm import tqdm
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=50,
    pred_iou_thresh=0.93,
    stability_score_thresh=0.94,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100)  # Requires open-cv to run post-processing)

def extract_polygons(segmentation, num_points_threshold, min_area):
    contours = measure.find_contours(segmentation, 0.5)
    contours = [contour for contour in contours if len(contour) >= num_points_threshold]
    contours = [contour for contour in contours if Polygon(contour).area >= min_area]
    polygons = []
    for contour in contours:
        polygon = []
        for point in contour:
            polygon.append(float(point[1]))  # X coordinate
            polygon.append(float(point[0]))  # Y coordinate
        polygons.append(polygon)
    return polygons

def save_results_to_json(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def process_images(image_dir, num_points_threshold, min_area):
    images = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            image = tiff.imread(image_path)
            band_red = image[:, :, 1]  
            band_green = image[:, :, 3] 
            band_blue = image[:, :, 6]  

            rgb_image = np.dstack((band_red, band_green, band_blue))
            image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
            segmentations = mask_generator.generate(np.array(image))  # Replace with actual function call to your segmentation model
            
            annotations = []
            for segmentation in segmentations:
                polygons = extract_polygons(segmentation['segmentation'], num_points_threshold, min_area)
                for polygon in polygons:
                    annotations.append({
                        "class": "field",
                        "segmentation": polygon
                    })
            images.append({
                "file_name": filename,
                "annotations": annotations
            })
    return {"images": images}


image_dir = 'competition\solafune\Field Area Segmentation\data\train_images'
output_file = 'results.json'
Number_of_point = 3
size = 10

# Process images and save results
results = process_images(image_dir, Number_of_point, size)
save_results_to_json(results, output_file)
print(f"Results saved to {output_file}")