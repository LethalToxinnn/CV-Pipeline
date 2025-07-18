# CV-Pipeline

A real-time 3D object detection system that combines YOLOv11 for object detection with Depth Anything v2 for depth estimation to create pseudo-3D bounding boxes and bird's eye view visualization.

## Features

- Real-time object detection using YOLOv11
- Depth estimation using Depth Anything v2
- 3D bounding box visualization
- Bird's Eye View (BEV) visualization
- Object tracking capabilities
- Support for video files and webcam input
- Adjustable model sizes for performance/accuracy tradeoffs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/LethalToxinnn/CV-Pipeline.git
   cd CV-Pipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download model weights (will be downloaded automatically on first run)

## Usage

Run the main script:

```bash
python run.py
```

### WebRTC Client Usage

For remote video processing from Raspberry Pi devices:

```bash
python client1.py
```

The WebRTC client connects to a WebSocket server (default: `ws://000.00.00.000:8080/ws`) and processes incoming video streams with 3D object detection in real-time.

### Configuration Options

You can modify the following parameters in `run.py`:

- **Input/Output**:
  - `source`: Path to input video file or webcam index (0 for default camera)
  - `output_path`: Path to output video file

- **Model Settings**:
  - `yolo_model_size`: YOLOv11 model size ("nano", "small", "medium", "large", "extra")
  - `depth_model_size`: Depth Anything v2 model size ("small", "base", "large")

- **Detection Settings**:
  - `conf_threshold`: Confidence threshold for object detection
  - `iou_threshold`: IoU threshold for NMS
  - `classes`: Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes

- **Feature Toggles**:
  - `enable_tracking`: Enable object tracking
  - `enable_bev`: Enable Bird's Eye View visualization
  - `enable_pseudo_3d`: Enable 3D visualization

## Project Structure

```
CV-Pipeline/

│── run.py                  # Main script
│── detection_model.py      # YOLOv11 object detection
│── depth_model.py          # Depth Anything v2 depth estimation
│── bbox3d_utils.py         # 3D bounding box utilities
│── load_camera_params.py   # Camera parameter utilities
├── requirements.txt        # Project dependencies
├── client1.py              # Runs webrtc client
└── README.md                   # This file
```

## How It Works

1. **Object Detection**: YOLOv11 detects objects in the frame and provides 2D bounding boxes
2. **Depth Estimation**: Depth Anything v2 generates a depth map for the entire frame
3. **3D Box Estimation**: Combines 2D boxes with depth information to create 3D boxes
4. **Visualization**: Renders 3D boxes and bird's eye view for better spatial understanding


## Changes from Original Repository

This repository is a fork of the original [YOLO-3D](https://github.com/niconielsen32/YOLO-3D.git) project with the following enhancements:

- **Repository Rename**: Changed from "YOLO-3D" to "CV-Pipeline" to better reflect the comprehensive computer vision pipeline
- **WebRTC Client**: Added `client1.py` for WebRTC-based real-time video streaming from Raspberry Pi devices, enabling remote 3D object detection processing
- **Enhanced Documentation**: Updated project structure and documentation to reflect new components

## Acknowledgments

- Original YOLO-3D project by [niconielsen32](https://github.com/niconielsen32/YOLO-3D.git)
- YOLOv11 by Ultralytics
- Depth Anything v2 by Microsoft

