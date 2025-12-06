#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ViTPose 推論腳本
輸入: dataset/images/ 裡的所有影格
輸出: output/keypoints_json/frame_xxxxxx.json
"""

import os
import mmcv
from mmpose.apis import (inference_top_down_pose_model,
                          init_pose_model,
                          vis_pose_result)
from mmpose.datasets import DatasetInfo
import argparse
import json
from glob import glob
from tqdm import tqdm

# -----------------------------
# 參數設定
# -----------------------------
parser = argparse.ArgumentParser(description='ViTPose 2D 推論')
parser.add_argument('--config', type=str, required=True, help='ViTPose config 檔案')
parser.add_argument('--checkpoint', type=str, required=True, help='訓練好的 checkpoint')
parser.add_argument('--input', type=str, required=True, help='影像資料夾')
parser.add_argument('--output', type=str, required=True, help='輸出 JSON 資料夾')
parser.add_argument('--vis', action='store_true', help='是否同時輸出可視化圖片')
parser.add_argument('--vis-out', type=str, default='output/vis', help='可視化圖片輸出資料夾')
args = parser.parse_args()

# -----------------------------
# 初始化模型
# -----------------------------
pose_model = init_pose_model(
    config=args.config,
    checkpoint=args.checkpoint,
    device='cuda:0'  # 如果沒有 GPU 改 'cpu'
)

dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
if dataset_info is not None:
    dataset_info = DatasetInfo(dataset_info)

# -----------------------------
# 建立輸出資料夾
# -----------------------------
os.makedirs(args.output, exist_ok=True)
if args.vis:
    os.makedirs(args.vis_out, exist_ok=True)

# -----------------------------
# 讀取所有影格
# -----------------------------
img_list = sorted(glob(os.path.join(args.input, '*.[jp][pn]g')))  # jpg/png
if len(img_list) == 0:
    raise ValueError(f"No images found in {args.input}")

# -----------------------------
# 推論每張影格
# -----------------------------
for img_path in tqdm(img_list, desc='Inference'):
    img_name = os.path.basename(img_path)
    
    # 只推論一個人 (Top-Down 模型)
    # bbox = [x, y, w, h], 可用整張圖
    import cv2
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    bbox = [[0, 0, w, h]]  # 整張影像當作一個 bbox

    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        img,
        bbox,
        bbox_score_thr=0.0,
        format='xywh',
        dataset_info=dataset_info
    )

    # -----------------------------
    # 存 JSON
    # -----------------------------
    out_file = os.path.join(args.output, os.path.splitext(img_name)[0] + '.json')
    with open(out_file, 'w') as f:
        json.dump(pose_results, f, indent=4)

    # -----------------------------
    # 可視化
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

print(f"Inference done! JSON results saved in {args.output}")
if args.vis:
    print(f"Visualization images saved in {args.vis_out}")