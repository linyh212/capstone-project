#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw keypoints on images based on ViTPose inference JSON results.
Input:
    --input-json : folder containing JSON results from infer.py
    --images     : folder containing original images
    --output     : folder to save visualized images
"""

import os
import argparse
import json
import cv2
from glob import glob
from tqdm import tqdm

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser(description='Draw keypoints on images')
parser.add_argument('--input-json', type=str, required=True,
                    help='Path to folder containing keypoints JSON files')
parser.add_argument('--images', type=str, required=True,
                    help='Path to folder containing original images')
parser.add_argument('--output', type=str, required=True,
                    help='Path to folder to save visualized images')
parser.add_argument('--radius', type=int, default=4, help='Keypoint circle radius')
parser.add_argument('--thickness', type=int, default=2, help='Line thickness')
args = parser.parse_args()

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs(args.output, exist_ok=True)

# -----------------------------
# Helper: draw keypoints
# -----------------------------
# skeleton connections for 12 joints (shoulder->elbow->wrist, hip->knee->ankle)
skeleton = [
    [0, 2], [2, 4],       # Left arm
    [1, 3], [3, 5],       # Right arm
    [0, 6], [1, 7],       # Torso (shoulder -> hip)
    [6, 8], [8, 10],      # Left leg
    [7, 9], [9, 11]       # Right leg
]

# color for keypoints and lines
kp_color = (0, 255, 0)  # green
line_color = (0, 0, 255)  # red

# -----------------------------
# Get all JSON files
# -----------------------------
json_files = sorted(glob(os.path.join(args.input_json, '*.json')))
if len(json_files) == 0:
    raise ValueError(f"No JSON files found in {args.input_json}")

# -----------------------------
# Process each JSON
# -----------------------------
for json_path in tqdm(json_files, desc='Drawing keypoints'):
    filename = os.path.basename(json_path)
    base_name = os.path.splitext(filename)[0]
    
    img_path_jpg = os.path.join(args.images, base_name + '.jpg')
    img_path_png = os.path.join(args.images, base_name + '.png')
    
    # load image
    if os.path.exists(img_path_jpg):
        img_path = img_path_jpg
    elif os.path.exists(img_path_png):
        img_path = img_path_png
    else:
        print(f"Image for {base_name} not found, skipped.")
        continue
    
    img = cv2.imread(img_path)
    
    # load keypoints
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # draw keypoints and skeleton
    for person in data:
        keypoints = person['keypoints']  # [x1, y1, score1, x2, y2, score2, ...]
        num_kpts = len(keypoints) // 3
        coords = []
        for i in range(num_kpts):
            x, y, v = keypoints[i*3:i*3+3]
            coords.append((int(x), int(y)))
            # draw keypoint circle if visible
            if v > 0:
                cv2.circle(img, (int(x), int(y)), args.radius, kp_color, -1)
        
        # draw skeleton lines
        for j1, j2 in skeleton:
            pt1 = coords[j1]
            pt2 = coords[j2]
            cv2.line(img, pt1, pt2, line_color, args.thickness)
    
    # save
    out_file = os.path.join(args.output, base_name + '.jpg')
    cv2.imwrite(out_file, img)

print(f"All images saved in {args.output}")