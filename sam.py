import numpy as np
import os
import json
from shapely.geometry import Polygon
import tifffile as tiff
from skimage import measure
from tqdm import tqdm
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint="sam/sam_hq_vit_h.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=100,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.85,
    crop_n_layers=2,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=20)  # Requires open-cv to run post-processing)


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

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def process_images(image_dir, num_points_threshold, min_area):
    images = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            image = tiff.imread(image_path)
            
            band_1 = image[:, :, 1]  
            band_2 = image[:, :, 3] 
            nir = image[:, :, 11]  
            
            ndvi = calculate_ndvi(nir, band_1)
            
            combined_image = np.dstack((band_1, band_2, ndvi))
            
            normalized_image = (combined_image - np.min(combined_image)) / (np.max(combined_image) - np.min(combined_image))
            segmentations = mask_generator.generate(np.array(normalized_image))
            
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

image_dir = 'field_area_segmentation/test_images/images'
output_file = 'results_base_b1_b2_ndvi.json'
Number_of_point = 3
size = 10

print("Starting image processing...")
results = process_images(image_dir, Number_of_point, size)
save_results_to_json(results, output_file)
print(f"Results saved to {output_file}")
print("Processing completed.")