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

def extract_polygons(segmentation):
    contours = measure.find_contours(segmentation, 0.5)
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
    return swir  # Return only the SWIR band

def calculate_pv_ir2(nir, swir1, swir2):
    return nir  # Return only the NIR band

def process_image(image_path, permutation, channel_names):
    image = tiff.imread(image_path)
    
    def get_channel(channel):
        if isinstance(channel, str):
            if channel == 'NDVI':
                nir = image[:, :, 7]  # B8
                red = image[:, :, 3]  # B4
                return calculate_ndvi(nir, red)
            elif channel == 'Urban_False_Color':
                swir = image[:, :, 10]  # B11
                nir = image[:, :, 7]   # B8
                red = image[:, :, 3]   # B4
                return calculate_urban_false_color(swir, nir, red)
            elif channel == 'PV-IR2':
                nir = image[:, :, 7]    # B8
                swir1 = image[:, :, 10] # B11
                swir2 = image[:, :, 11] # B12
                return calculate_pv_ir2(nir, swir1, swir2)
            else:
                return image[:, :, channel_names.index(channel)]
        else:
            return image[:, :, channel]

    channels = [get_channel(channel) for channel in permutation]
    processed_image = np.dstack(channels)
    
    normalized_image = (processed_image - np.min(processed_image)) / (np.max(processed_image) - np.min(processed_image))
    
    segmentations = mask_generator.generate(np.array(normalized_image)) 
    annotations = []
    for segmentation in segmentations:
        polygons = extract_polygons(segmentation['segmentation'])
        for polygon in polygons:
            annotations.append({
                "class": "field",
                "segmentation": polygon
            })
    return {
        "file_name": os.path.basename(image_path),
        "annotations": annotations
    }

def process_images(image_dir, output_dir):
    channel_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    special_indices = ['NDVI', 'Urban_False_Color', 'PV-IR2']
    
    # Generate all possible 3-channel combinations
    channel_permutations = list(itertools.combinations(range(12), 3))
    
    # Create permutations including special indices
    all_permutations = set()  # Use a set to avoid duplicates
    for perm in channel_permutations:
        all_permutations.add(perm)
        for special in special_indices:
            all_permutations.add((perm[0], perm[1], special))
    
    # Add special indices combinations
    all_permutations.update(itertools.combinations(special_indices, 3))
    
    # Get existing result files
    existing_results = set(os.listdir(output_dir))
    
    # Get all .tif files and sort them
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    
    # Select specific images (1st, 15th, 25th)
    selected_images = [all_images[i] for i in [0, 14, 24] if i < len(all_images)]
    
    total_permutations = len(all_permutations)
    processed_permutations = 0
    
    for permutation in all_permutations:
        perm_name = "-".join([channel_names[i] if isinstance(i, int) else i for i in permutation])
        output_file = f'segmentation_results_{perm_name}.json'
        
        if output_file in existing_results:
            print(f"Skipping existing permutation: {perm_name}")
            processed_permutations += 1
            continue
        
        print(f"\nProcessing permutation: {perm_name} ({processed_permutations + 1}/{total_permutations})")
        
        results = {"images": []}
        for filename in tqdm(selected_images):
            image_path = os.path.join(image_dir, filename)
            image_result = process_image(image_path, permutation, channel_names)
            results["images"].append(image_result)
        
        output_path = os.path.join(output_dir, output_file)
        save_results_to_json(results, output_path)
        print(f"Results saved to {output_path}")
        
        processed_permutations += 1

    print(f"Processing completed. Total permutations processed: {processed_permutations}/{total_permutations}")

# Directory containing images
image_dir = '/home/protostartserver2/Public/field_area_segmentation/train_images/images'
output_dir = '/home/protostartserver2/Public/field_area_segmentation/testing_channels'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

print("Starting image processing...")
process_images(image_dir, output_dir)
print("Processing completed.")