#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ViTPose 2D Inference Script
Input: All frames inside dataset/images/
Output: JSON files saved to output/keypoints_json/frame_xxxxxx.json
"""

import os
import mmcv
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    vis_pose_result
)
from mmpose.datasets import DatasetInfo
import argparse
import json
from glob import glob
from tqdm import tqdm

# -----------------------------
# Argument Settings
# -----------------------------
parser = argparse.ArgumentParser(description='ViTPose 2D Inference')
parser.add_argument('--config', type=str, required=True,
                    help='Path to ViTPose config file')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to trained checkpoint')
parser.add_argument('--input', type=str, required=True,
                    help='Folder containing input images')
parser.add_argument('--output', type=str, required=True,
                    help='Folder to store output JSON files')
parser.add_argument('--vis', action='store_true',
                    help='Enable output of visualization images')
parser.add_argument('--vis-out', type=str, default='output/vis',
                    help='Folder to store visualization results')
args = parser.parse_args()

# -----------------------------
# Initialize Model
# -----------------------------
pose_model = init_pose_model(
    config=args.config,
    checkpoint=args.checkpoint,
    device='cuda:0'  # Change to 'cpu' if no GPU available
)

dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
if dataset_info is not None:
    dataset_info = DatasetInfo(dataset_info)

# -----------------------------
# Create Output Folders
# -----------------------------
os.makedirs(args.output, exist_ok=True)
if args.vis:
    os.makedirs(args.vis_out, exist_ok=True)

# -----------------------------
# Load All Frames
# -----------------------------
img_list = sorted(glob(os.path.join(args.input, '*.[jp][pn]g')))  # jpg/png
if len(img_list) == 0:
    raise ValueError(f"No images found in {args.input}")

# -----------------------------
# Inference on Each Frame
# -----------------------------
for img_path in tqdm(img_list, desc='Inference'):
    img_name = os.path.basename(img_path)

    # Use full image as a single bounding box (Top-Down inference)
    import cv2
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    bbox = [[0, 0, w, h]]  # Whole image

    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        img,
        bbox,
        bbox_score_thr=0.0,
        format='xywh',
        dataset_info=dataset_info
    )

    # -----------------------------
    # Save JSON Output
    # -----------------------------
    out_file = os.path.join(args.output, os.path.splitext(img_name)[0] + '.json')
    with open(out_file, 'w') as f:
        json.dump(pose_results, f, indent=4)

    # -----------------------------
    # Visualization Output
    # -----------------------------
    if args.vis:
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset_info=dataset_info,
            kpt_score_thr=0.3,
            radius=4,
            thickness=2,
            show=False
        )
        cv2.imwrite(os.path.join(args.vis_out, img_name), vis_img)

print(f"Inference complete! JSON results saved in {args.output}")
if args.vis:
    print(f"Visualization images saved in {args.vis_out}")