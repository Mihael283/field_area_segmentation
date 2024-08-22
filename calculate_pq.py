import numpy as np
import os
import json
from shapely.geometry import Polygon
import pandas as pd
import itertools
from tqdm import tqdm
import sys

def getIOU(polygon1: Polygon, polygon2: Polygon):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    if union == 0:
        return 0
    return intersection / union

def compute_pq(gt_polygons: list, pred_polygons: list, iou_threshold=0.5):
    matched_instances = {}
    gt_matched = np.zeros(len(gt_polygons))
    pred_matched = np.zeros(len(pred_polygons))
    
    for gt_idx, gt_polygon in enumerate(gt_polygons):
        best_iou = iou_threshold
        best_pred_idx = None
        for pred_idx, pred_polygon in enumerate(pred_polygons):
            try:
                iou = getIOU(gt_polygon, pred_polygon)
            except:
                iou = 0
                print('Error Polygon -> iou is 0')
            
            if iou == 0:
                continue
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_instances[(gt_idx, best_pred_idx)] = best_iou
            gt_matched[gt_idx] = 1
            pred_matched[best_pred_idx] = 1

    sq_sum = sum(matched_instances.values())
    num_matches = len(matched_instances)
    sq = sq_sum / num_matches if num_matches else 0
    rq = num_matches / ((len(gt_polygons) + len(pred_polygons))/2.0) if (gt_polygons or pred_polygons) else 0
    pq = sq * rq

    return pq, sq, rq

with open('train_annotation.json') as f:
    gts = json.load(f)

for k in [k for k in os.listdir('testing_channels')]:
    csv_filename = f'./testing_csvs/exp_{k}.csv'
    
    # Check if CSV already exists
    if os.path.exists(csv_filename):
        print(f"CSV for {k} already exists. Skipping computation.")
        continue
    
    print(f"Processing {k}")
    with open(f'testing_channels/{k}') as f:
        submits_json = json.load(f)
    
    scores, files = [], []
    for i, (_image_pred) in tqdm(enumerate(submits_json['images'])):
        fname = _image_pred['file_name']
        annos_pred = _image_pred['annotations']
        
        for j, (_image_gt) in enumerate(gts['images']):
            if _image_gt['file_name'] == fname:
                fname_gt = _image_gt['file_name']
                break
        
        annos_gt = _image_gt['annotations']
        
        print(f'File:{fname} - {fname_gt} Num GT: {len(annos_gt)}, Num Pred: {len(annos_pred)}')
        
        polygons_gt, polygons_pred = [], []
        for anno in annos_gt:
            _polys = []
            for ii, (x, y) in enumerate(zip(anno['segmentation'][::2], anno['segmentation'][1::2])):
                _polys.append((x, y))
            polygons_gt.append(Polygon(_polys))
            
        for anno in annos_pred:
            _polys = []
            for ii, (x, y) in enumerate(zip(anno['segmentation'][::2], anno['segmentation'][1::2])):
                _polys.append((x, y))
            polygons_pred.append(Polygon(_polys))
        
        pq, sq, rq = compute_pq(polygons_gt, polygons_pred)
        print(f'File:{fname} PQ: {pq:.4f}, SQ: {sq:.4f}, RQ: {rq:.4f} Num: {len(polygons_gt)}')
        
        scores.append([pq, sq, rq])
        files.append(fname)
    
    df = pd.DataFrame(scores, columns=['PQ', 'SQ', 'RQ'], index=files)
    df['file'] = files

    metrics = df['PQ'].mean()
    print(f'Mean PQ: {metrics:.4f}')
    df.to_csv(csv_filename)