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

model_type = "vit_b"
device = "cuda:1"
sam = sam_model_registry[model_type](checkpoint="sam/sam_vit_b_01ec64.pth")
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

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def calculate_urban_false_color(swir, nir, red):
    return np.dstack((swir, nir, red))

def calculate_pv_ir2(nir, swir1, swir2):
    return np.dstack((nir, swir1, swir2))

def process_images(image_dir, num_points_threshold, min_area, output_file):
    images = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')][:4]
    total_images = len(image_files)
    
    channel_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    for img_idx, filename in enumerate(image_files, 1):
        print(f"\nProcessing image {img_idx}/{total_images}: {filename}")
        
        image_path = os.path.join(image_dir, filename)
        image = tiff.imread(image_path)
        
        channel_combinations = list(itertools.combinations(range(12), 3))
        
        permutations = []
        for combo in channel_combinations:
            combo_name = f"{channel_names[combo[0]]}-{channel_names[combo[1]]}-{channel_names[combo[2]]}"
            combo_image = np.dstack((image[:,:,combo[0]], image[:,:,combo[1]], image[:,:,combo[2]]))
            permutations.append((combo_name, combo_image))
        
        #NDVI
        ndvi = calculate_ndvi(image[:,:,7], image[:,:,3])  # B8 (NIR) and B4 (Red)
        permutations.append(('NDVI', np.dstack((ndvi, ndvi, ndvi))))
        
        #Urban False Color
        urban_false_color = calculate_urban_false_color(image[:,:,10], image[:,:,7], image[:,:,3])  # B11 (SWIR1), B8 (NIR), B4 (Red)
        permutations.append(('Urban False Color', urban_false_color))
        
        #PV-IR2
        pv_ir2 = calculate_pv_ir2(image[:,:,7], image[:,:,10], image[:,:,11])  # B8 (NIR), B11 (SWIR1), B12 (SWIR2)
        permutations.append(('PV-IR2', pv_ir2))
        
        total_permutations = len(permutations)
        annotations = []
        
        for perm_idx, (index_name, image_permutation) in enumerate(permutations, 1):
            print(f"  Processing permutation {perm_idx}/{total_permutations}: {index_name}")
            
            image_normalized = (image_permutation - np.min(image_permutation)) / (np.max(image_permutation) - np.min(image_permutation))
            segmentations = mask_generator.generate(np.array(image_normalized))
            
            for segmentation in segmentations:
                polygons = extract_polygons(segmentation['segmentation'], num_points_threshold, min_area)
                for polygon in polygons:
                    annotations.append({
                        "class": "field",
                        "segmentation": polygon,
                        "index": index_name
                    })
        
        images.append({
            "file_name": filename,
            "annotations": annotations
        })
        
        interim_results = {"images": images}
        save_results_to_json(interim_results, output_file)
        print(f"Interim results saved after processing image {img_idx}/{total_images}: {filename}")
    
    return {"images": images}

image_dir = '/home/protostartserver2/Public/field_area_segmentation/test_images/images'
output_file = 'test_sam.json'
Number_of_point = 3
size = 10

print("Starting image processing...")
results = process_images(image_dir, Number_of_point, size, output_file)
print(f"Final results saved to {output_file}")
print("Processing completed.")