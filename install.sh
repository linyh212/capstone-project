#!/usr/bin/env bash

# ===========================
# ViTPose 2D pipeline (Bash shell)
# 1. 影片拆影格
# 2. 聚集影格到 dataset/images
# 3. 訓練 ViTPose
# 4. 推論
# 5. 畫骨架 keypoints
# 6. 合成最終影片
# ===========================

set -e  # 出現錯誤就停止

# ---------------------------
# Step 0: Create base folders
# ---------------------------
mkdir -p frames
mkdir -p dataset/images
mkdir -p work_dirs/vitpose_run1
mkdir -p output/keypoints_json
mkdir -p output/vis

# ---------------------------
# Step 1: extract frames
# ---------------------------
echo "=== Step 1: extract frames ==="
for f in videos/*.{mp4,MP4}; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .mp4)
        mkdir -p "frames/$name"
        echo "Extracting frames from $f ..."
        ffmpeg -i "$f" -vf "fps=30" "frames/$name/frame_%06d.jpg"
    fi
done

# ---------------------------
# Step 2: gather frames into dataset/images + reorder filenames
# ---------------------------
echo "=== Step 2: gather frames into dataset/images and rename ==="

# 清空 dataset/images 以防舊檔干擾
rm -f dataset/images/*.jpg

counter=1
for d in frames/*; do
    if [ -d "$d" ]; then
        for img in "$d"/*.jpg; do
            # 新檔名 frame_000001.jpg, frame_000002.jpg ...
            printf -v newname "frame_%06d.jpg" "$counter"
            cp "$img" "dataset/images/$newname"
            ((counter++))
        done
    fi
done

echo "Total frames copied and renamed: $((counter-1))"

# ---------------------------
# Step 3: training ViTPose
# ---------------------------
echo "=== Step 3: training ViTPose ==="
mim train mmpose configs/vitpose_custom.py --work-dir work_dirs/vitpose_run1

# ---------------------------
# Step 4: inference
# ---------------------------
echo "=== Step 4: inference ==="
python scripts/infer.py \
    --config configs/vitpose_custom.py \
    --checkpoint work_dirs/vitpose_run1/latest.pth \
    --input dataset/images \
    --output output/keypoints_json

# ---------------------------
# Step 5: draw keypoints
# ---------------------------
echo "=== Step 5: draw keypoints ==="
python scripts/draw_keypoints.py \
    --input-json output/keypoints_json \
    --images dataset/images \
    --output output/vis

# ---------------------------
# Step 6: make final video
# ---------------------------
echo "=== Step 6: make final video ==="
ffmpeg -framerate 30 -pattern_type glob -i "output/vis/*.jpg" -c:v libx264 -pix_fmt yuv420p output/final.mp4

echo "=== Pipeline finished! ==="
