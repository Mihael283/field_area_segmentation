import json
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

def load_ndvi(image_path):
    with rasterio.open(image_path) as src:
        red = src.read(4).astype(float)
        nir = src.read(8).astype(float)
        ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def segmentation_to_polygon(segmentation):
    coords = np.array(segmentation).reshape(-1, 2)
    return Polygon(coords)

def draw_polygon(draw, polygon, outline_color, fill_color):
    draw.polygon(list(polygon.exterior.coords), outline=outline_color, fill=fill_color)

def visualize_result(image_path, annotations, output_path):
    # Load NDVI
    ndvi = load_ndvi(image_path)
    
    # Normalize NDVI for visualization
    ndvi_normalized = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))
    
    # Create RGB image from NDVI
    ndvi_rgb = (ndvi_normalized * 255).astype(np.uint8)
    ndvi_image = Image.fromarray(ndvi_rgb)
    ndvi_image = ndvi_image.convert("RGB")
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', ndvi_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw polygons
    for annotation in annotations:
        polygon = segmentation_to_polygon(annotation['segmentation'])
        draw_polygon(draw, polygon, outline_color=(255, 0, 0), fill_color=(255, 0, 0, 64))
    
    # Combine images
    combined = Image.alpha_composite(ndvi_image.convert('RGBA'), overlay)
    
    # Save the result
    combined.save(output_path)

def main():
    # Load results
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    # Create output directory
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each result
    for image_result in results['images']:
        file_name = image_result['file_name']
        image_path = os.path.join('test_images/images', file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_result.png")
        
        visualize_result(image_path, image_result['annotations'], output_path)
        print(f"Visualization saved for {file_name}")

if __name__ == "__main__":
    main()