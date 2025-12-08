#!/usr/bin/env bash

# ViTPose 2D Pipeline (Bash Shell)
# 1. Extract frames from videos
# 2. Collect frames into dataset/images
# 3. Train ViTPose
# 4. Run inference
# 5. Draw skeleton keypoints
# 6. Generate final output video

set -e  # Stop the script if any command fails

# ---------------------------
# 0) Create required directories
# ---------------------------
mkdir -p frames
mkdir -p dataset/images
mkdir -p work_dirs/vitpose_run1
mkdir -p output/keypoints_json
mkdir -p output/vis

# ---------------------------
# 1) Extract frames
# ---------------------------
echo "=== Step 1: Extracting frames ==="
for f in videos/*.{mp4,MP4}; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .mp4)
        mkdir -p "frames/$name"
        echo "Extracting frames from $f ..."
        ffmpeg -i "$f" -vf "fps=30" "frames/$name/frame_%06d.jpg"
    fi
done

# ---------------------------
# 2) Gather frames into dataset/images and rename
# ---------------------------
echo "=== Step 2: Gathering and renaming frames ==="

# Clear existing images to avoid conflicts
rm -f dataset/images/*.jpg

counter=1
for d in frames/*; do
    if [ -d "$d" ]; then
        for img in "$d"/*.jpg; do
            # Rename to frame_000001.jpg, frame_000002.jpg, ...
            printf -v newname "frame_%06d.jpg" "$counter"
            cp "$img" "dataset/images/$newname"
            ((counter++))
        done
    fi
done

echo "Total frames copied and renamed: $((counter-1))"

# ---------------------------
# 3) Train ViTPose
# ---------------------------
echo "=== Step 3: Training ViTPose ==="
mim train mmpose configs/vitpose_custom.py --work-dir work_dirs/vitpose_run1

# ---------------------------
# 4) Inference
# ---------------------------
echo "=== Step 4: Running inference ==="
python scripts/infer.py \
    --config configs/vitpose_custom.py \
    --checkpoint work_dirs/vitpose_run1/latest.pth \
    --input dataset/images \
    --output output/keypoints_json

# ---------------------------
# 5) Draw keypoints
# ---------------------------
echo "=== Step 5: Drawing keypoints ==="
python scripts/draw_keypoints.py \
    --input-json output/keypoints_json \
    --images dataset/images \
    --output output/vis

# ---------------------------
# 6) Generate final video
# ---------------------------
echo "=== Step 6: Generating final video ==="
ffmpeg -framerate 30 -pattern_type glob -i "output/vis/*.jpg" \
    -c:v libx264 -pix_fmt yuv420p output/final.mp4

echo "=== Pipeline completed successfully! ==="