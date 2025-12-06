# Dragon Boat Motion Analysis Project

## Project Overview
This capstone project focuses on **motion analysis for dragon boat athletes** using **computer vision and deep learning**.  
The main goal is to implement **Human Pose Estimation** and **Video → 2D → 3D model transformation** using **video-based detection and tracking**.

The final project pipeline includes:

- **Paddle detection** — detect and track the paddle in the video.
- **2D Human Pose Estimation** — using ViTPose to extract body keypoints.
- **3D Human Pose Reconstruction** — reconstruct 3D joint positions from 2D keypoints.
- **Data Analysis** — extract stroke frequency, stroke distance, entry/exit angles, velocity, and other motion metrics.

This workflow supports **coaches, athletes, and researchers** in analyzing paddling techniques more precisely.

---

## Repository Structure
* `configs` — includings `vitpose_custom.py`
* `scripts` — includings `draw_keypoints.py`, `infer.py`
* `install.sh` — 

---

## Model Development

### 1. Human Detection

### 2. 2D Human Pose Estimation
- **ViTPose** is used for **top-down 2D pose estimation**.
- Only **12 keypoints below the head** are labeled:
  - Left/Right Shoulder, Elbow, Wrist
  - Left/Right Hip, Knee, Ankle
- Pipeline includes:
  1. Extract frames from input videos
  2. Prepare dataset in COCO format
  3. Train ViTPose on custom 12-joint dataset
  4. Inference on all frames
  5. Visualize keypoints
  6. Assemble frames into final video

### 3. 3D Human Pose Reconstruction
- Future step: Convert 2D keypoints to **3D skeleton** using methods like **VideoPose3D**.
- Enables measurement of stroke angles, joint trajectories, and velocities.

---

## Setup Instructions

1. **Clone repository**
```bash
git clone https://github.com/linyh212/capstone-project.git
cd ~/capstone-project
```

2. **Prepare dataset**
```bash
mkdir videos
```
    •	Place videos into videos/
    •	Frames will be automatically extracted and copied to dataset/images/.

3. **Install dependencies and start `video to 2D` process**
```bash
bash install.sh
```

# Research Background
Reference material from the **Sports Science & Technology Center (運動科學與科技中心)** presentation outlines real-world metrics in dragon boat racing, including:

* Stroke frequency (spm)
* Stroke distance (cm)
* Entry, exit, and maximum angles (degrees)
* Stroke duration and recovery time (ms)

The model aims to extract similar measurements **automatically from video footage**.
