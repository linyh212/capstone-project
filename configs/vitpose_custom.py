# vitpose_custom.py
# Purpose: Train a ViTPose 2D model using "12 lower-body keypoints" (shoulder and below)
# Dataset structure you specified:
#   dataset/annotations/train.json
#   dataset/annotations/val.json   (same format as train)
#   dataset/images/                (all extracted frames)

_base_ = [
    # Base config for ViTPose (COCO / top-down heatmap)
    '../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py',
]

custom_imports = dict(imports=['mmpretrain'], allow_failed_imports=False)

# -----------------------------
# 1) Dataset (pointing to dataset/annotations and dataset/images)
# -----------------------------
# In mmpose, data_root is the parent directory of both annotations and images.
# data_prefix specifies the subdirectory for images.
train_dataloader = dict(
    dataset=dict(
        data_root='dataset/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root='dataset/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
    )
)
test_dataloader = val_dataloader

# -----------------------------
# 2) Keypoint definition (12 keypoints below the head)
#    IMPORTANT: Your annotation JSON must use the same order + count (12 pts)
# -----------------------------
# Keypoint order (fixed):
#  1 LShoulder,  2 RShoulder,
#  3 LElbow,     4 RElbow,
#  5 LWrist,     6 RWrist,
#  7 LHip,       8 RHip,
#  9 LKnee,     10 RKnee,
# 11 LAnkle,    12 RAnkle
keypoint_names = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

# Skeleton (useful for visualization/evaluation; not required for training)
skeleton = [
    [1, 3], [3, 5],          # Left arm
    [2, 4], [4, 6],          # Right arm
    [1, 7], [2, 8],          # Torso (shoulder -> hip)
    [7, 9], [9, 11],         # Left leg
    [8, 10], [10, 12],       # Right leg
]

# -----------------------------
# 3) Overwrite model output dimension (CRITICAL: out_channels = 12)
# -----------------------------
# This makes ViTPose predict 12 heatmaps instead of COCO’s 17
model = dict(
    head=dict(
        out_channels=12
    )
)

# -----------------------------
# 4) Overwrite dataset configuration (num_keypoints = 12)
#    NOTE: In some mmpose versions, data_cfg lives in different locations.
#    If you still see "num_keypoints = 17" during training,
#    keep this section and ensure it overrides the base config.
# -----------------------------
data_cfg = dict(
    num_keypoints=12
)

# -----------------------------
# 5) (Optional) Training hyperparameters
# -----------------------------
train_cfg = dict(
    max_epochs=210
)

# If your dataset is small, reduce batch size depending on GPU memory
train_dataloader.update(dict(batch_size=16))
