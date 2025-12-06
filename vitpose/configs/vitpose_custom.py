# vitpose_custom.py
# 目的：在你的專案中訓練「只含身體（肩膀以下）12 個 keypoints」的 ViTPose 2D 模型
# 資料位置（你指定的專案結構）：
#   dataset/annotations/train.json
#   dataset/annotations/val.json   (與 train.json 相同)
#   dataset/images/              (所有影格影像)

_base_ = [
    'mmpose/configs/_base_/default_runtime.py',
    'mmpose/configs/_base_/datasets/coco.py',
    # ViTPose 的常用基底 config（COCO / topdown heatmap）
    'mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py',
]

# -----------------------------
# 1) dataset (指向你的 dataset/annotations + dataset/images)
# -----------------------------
train_dataloader = dict(
    dataset=dict(
        ann_file='dataset/annotations/train.json',
        data_root='dataset/images/'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='dataset/annotations/val.json',
        data_root='dataset/images/'
    )
)

# （可選）如果你之後要用 test 運行，也可以同步指向同一套資料
test_dataloader = val_dataloader

# -----------------------------
# 2) keypoints 定義（只保留肩膀以下 12 點）
#    你的人為標註 JSON 必須用同一順序與數量（12 points）
# -----------------------------
# 12 points 順序（固定）：
# 1 LShoulder, 2 RShoulder, 3 LElbow, 4 RElbow, 5 LWrist, 6 RWrist,
# 7 LHip,      8 RHip,      9 LKnee, 10 RKnee, 11 LAnkle, 12 RAnkle
keypoint_names = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

# skeleton 可用於可視化/評估（不是訓練必須，但建議補齊）
skeleton = [
    [1, 3], [3, 5],          # L arm
    [2, 4], [4, 6],          # R arm
    [1, 7], [2, 8],          # torso (shoulder -> hip)
    [7, 9], [9, 11],         # L leg
    [8, 10], [10, 12],       # R leg
]

# -----------------------------
# 3) 覆寫模型輸出維度（非常關鍵：out_channels = 12）
# -----------------------------
# 這會把 ViTPose 的 keypoint head 改成輸出 12 個 heatmap（對應 12 個關節）
model = dict(
    keypoint_head=dict(
        out_channels=12
    )
)

# -----------------------------
# 4) 覆寫資料配置（把 num_keypoints 改成 12）
#    注意：不同版本的 base config 可能把 data_cfg 放在不同位置；
#    如果你訓練時看到 "num_keypoints" 仍是 17，代表你的安裝版本把它放在別的 key，
#    那麼你只要把下面這段（data_cfg）保留，並確保它在 config 最後被覆蓋到。
# -----------------------------
data_cfg = dict(
    num_keypoints=12
)

# -----------------------------
# 5)（可選）訓練超參數微調（你可先用預設，後面再調）
# -----------------------------
train_cfg = dict(
    max_epochs=210
)

# 如果你的資料量不大，建議先把 batch size 準到你的 GPU 能吃：
train_dataloader.update(dict(batch_size=16))