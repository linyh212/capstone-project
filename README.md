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

## Setup and Installation

### Prerequisites
Before you begin, ensure you have the following installed on your system:
- [Git](https://git-scm.com/downloads)
- [Conda (Anaconda or Miniconda)](https://docs.conda.io/en/latest/miniconda.html)
- [ffmpeg](https://ffmpeg.org/download.html): A command-line tool for handling video and audio. You can install it via Homebrew (`brew install ffmpeg`) on macOS, or follow the official instructions for your operating system.

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/linyh212/capstone-project.git
   cd capstone-project
   ```

2. **Create and Activate Conda Environment**
   The `environment.yaml` file defines *all Python dependencies* for this project, including PyTorch, OpenMMLab libraries (MMPose, MMDetection, MMPretrain, MMCV, MMEngine), and other utilities.

   Use the provided `environment.yaml` file to create a new conda environment:
   ```bash
   conda env create -f environment.yaml
   ```
   This will create a new environment named `pose`. Activate it:
   ```bash
   conda activate pose
   ```
   **Note:** The `environment.yaml` was generated on a macOS ARM64 system. If you are on a different architecture (e.g., x86_64 Linux/Windows) and encounter issues, you might need to:
   - Manually adjust the PyTorch installation command based on your CUDA version and OS.
   - Install specific versions of OpenMMLab packages that are compatible with your system. During development, we found the following compatible versions:
     - `mmpose>=1.0.0` (specifically tested with `1.3.2`)
     - `mmdet==3.2.0`
     - `mmcv==2.1.0`
     - `mmpretrain`

   - Alternatively, you can create an empty conda environment and manually install dependencies.

3. **Clone the MMPose Repository**
   The project's configuration file (`configs/vitpose_custom.py`) inherits from MMPose's base configurations. Therefore, you need to clone the `mmpose` repository into the project root:
   ```bash
   git clone https://github.com/open-mmlab/mmpose.git
   ```

---

## Usage

### 1. Prepare Your Dataset

- **Videos**: Create a `videos/` directory in the project root and place your input videos (e.g., `.mp4` files) inside it.
- **Annotations (for training)**: The training process requires annotation files in the COCO format.
    - Create a directory `dataset/annotations/`.
    - Place your `train.json` and `val.json` files inside `dataset/annotations/`. These files must contain the 12 lower-body keypoints defined in `configs/vitpose_custom.py`.

### 2. Run the Full Pipeline
The `install.sh` script automates the entire workflow from video processing to model training and inference.

Make sure your `pose` conda environment is activated, then run:
```bash
bash install.sh
```

This script will:
1.  **Create Directories**: Set up the necessary folder structure (`frames`, `dataset/images`, `work_dirs`, `output`).
2.  **Extract Frames**: Use `ffmpeg` to convert your videos into image frames.
3.  **Gather Frames**: Collect all frames into the `dataset/images` directory.
4.  **Train ViTPose**: Train the pose estimation model using your custom dataset.
5.  **Run Inference**: Perform pose estimation on the frames and generate JSON files with keypoint data.
6.  **Draw Keypoints**: Visualize the results by drawing skeletons on the frames.
7.  **Generate Final Video**: Create a final `.mp4` video from the visualized frames.

---

## Repository Structure
* `configs/vitpose_custom.py`: Configuration file for training the ViTPose model. Defines dataset paths, keypoint definitions, and model hyperparameters.
* `scripts/`: Contains Python scripts for individual pipeline steps.
    * `draw_keypoints.py`: Visualizes the pose estimation results.
    * `infer.py`: Runs inference with a trained model.
* `install.sh`: The main executable script that orchestrates the entire pipeline.
* `environment.yaml`: Conda environment file for easy setup.
* `info/`: Contains project-related documents.

---

## Model Development

### 1. Human Detection
- **Purpose:** Identify human regions (bounding boxes) in each frame.  
- **Tool:** ViTPose top-down pipeline uses a **pretrained detector** (e.g., Faster R-CNN) to locate humans.  
- **Note:** Bounding boxes are used to crop input for 2D pose estimation.

### 2. 2D Human Pose Estimation
- **Purpose:** Detect body keypoints to form a skeleton.  
- **Tool:** **ViTPose** (inside **MMpose** framework) performs **top-down 2D pose estimation**.
- **Keypoints:** 12 joints below the head:
  - Left/Right Shoulder, Elbow, Wrist
  - Left/Right Hip, Knee, Ankle
- **Pipeline:**
  1. Extract frames from input videos.
  2. Gather frames into `dataset/images/` with **unique sequential filenames**.
  3. Train ViTPose on **custom 12-joint COCO-style dataset**.
  4. Run inference on all frames → output JSON keypoints.
  5. Visualize keypoints using `draw_keypoints.py`.
  6. Assemble frames into final video using ffmpeg.

### 3. 3D Human Pose Reconstruction
- **Purpose:** Convert 2D keypoints to **3D skeleton**.
- **Tool:** Methods like **VideoPose3D** can estimate 3D joint positions.
- **Output:** Enables calculation of stroke angles, joint trajectories, velocities, etc.

---

## Research Background
Reference material from the **Sports Science & Technology Center (運動科學與科技中心)** presentation outlines real-world metrics in dragon boat racing, including:

* Stroke frequency (spm)
* Stroke distance (cm)
* Entry, exit, and maximum angles (degrees)
* Stroke duration and recovery time (ms)

The model aims to extract similar measurements **automatically from video footage**.
