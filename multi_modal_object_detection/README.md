# Multi-Modal Object Detection for Autonomous Driving

A comprehensive object detection system that fuses data from multiple sensors (Radar, LiDAR, and Camera) to detect and annotate objects for autonomous driving models.

## 🎯 Overview

This application combines three sensor modalities to provide robust object detection:

- **Camera**: RGB image-based 2D detection with visual features
- **LiDAR**: 3D point cloud-based detection with depth information
- **Radar**: Range and velocity data for moving object detection

## 🏗️ Project Structure

```
multi_modal_object_detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── config/
│   └── config.yaml                    # Configuration file
├── src/
│   ├── __init__.py
│   ├── camera_detector.py             # Camera-based detection
│   ├── lidar_detector.py              # LiDAR-based detection
│   ├── radar_detector.py              # Radar-based detection
│   ├── fusion_engine.py               # Multi-modal fusion
│   ├── data_processor.py              # Data loading and preprocessing
│   └── visualizer.py                  # Visualization tools
├── examples/
│   ├── example_camera.py              # Camera detection example
│   ├── example_lidar.py               # LiDAR detection example
│   ├── example_radar.py               # Radar detection example
│   ├── example_fusion.py              # Full sensor fusion example
│   └── sample_data/
│       ├── camera_image.jpg
│       ├── lidar_pointcloud.pcd
│       └── radar_data.json
├── models/
│   └── yolov8n.pt                     # Pre-trained YOLOv8 model
├── output/
│   ├── annotated_images/
│   ├── annotated_pointclouds/
│   └── fusion_results.json
└── tests/
    └── test_detectors.py              # Unit tests
```

## 🚀 Features

- **Multi-Sensor Fusion**: Combines detections from camera, LiDAR, and radar
- **Real-time Processing**: Efficiently processes streaming sensor data
- **3D Bounding Boxes**: Generates 3D object annotations
- **Confidence Scoring**: Provides confidence scores for detections
- **Data Annotation**: Outputs ready-to-use annotated data in multiple formats
- **Visualization**: Visual tools for debugging and analysis

## 📦 Installation

1. Clone the repository:
```bash
cd ..
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Quick Start

### Basic Camera Detection
```python
from src.camera_detector import CameraDetector

detector = CameraDetector()
results = detector.detect("examples/sample_data/camera_image.jpg")
detector.visualize_results(results)
```

### LiDAR Point Cloud Detection
```python
from src.lidar_detector import LiDARDetector

detector = LiDARDetector()
results = detector.detect("examples/sample_data/lidar_pointcloud.pcd")
detector.visualize_results(results)
```

### Radar Detection
```python
from src.radar_detector import RadarDetector

detector = RadarDetector()
results = detector.detect("examples/sample_data/radar_data.json")
detector.print_detections(results)
```

### Multi-Modal Sensor Fusion
```python
from src.fusion_engine import FusionEngine

fusion = FusionEngine()
results = fusion.process(
    camera_data="examples/sample_data/camera_image.jpg",
    lidar_data="examples/sample_data/lidar_pointcloud.pcd",
    radar_data="examples/sample_data/radar_data.json"
)
fusion.visualize_fusion_results(results)
```

## 🔄 How It Works

### 1. **Camera Detection (2D)**
- Uses YOLOv8 for real-time object detection
- Detects: cars, pedestrians, cyclists, trucks, etc.
- Outputs: 2D bounding boxes with class labels and confidence

### 2. **LiDAR Detection (3D)**
- Processes 3D point cloud data
- Generates 3D bounding boxes in XYZ coordinates
- Provides depth information and object height

### 3. **Radar Detection (Velocity)**
- Extracts range, angle, and doppler velocity
- Identifies moving objects
- Provides relative velocity information

### 4. **Sensor Fusion**
- Matches detections across modalities
- Combines 2D and 3D information
- Resolves conflicts with confidence scoring
- Outputs unified annotation format

## 📊 Output Formats

### Annotated Image (Camera)
```
{
  "image": "path/to/image.jpg",
  "detections": [
    {
      "class": "car",
      "bbox_2d": [x1, y1, x2, y2],
      "confidence": 0.95,
      "color": [0, 255, 0]
    }
  ]
}
```

### Annotated Point Cloud (LiDAR)
```
{
  "pointcloud": "path/to/pointcloud.pcd",
  "detections": [
    {
      "class": "car",
      "bbox_3d": {
        "center": [x, y, z],
        "size": [length, width, height],
        "rotation": angle
      },
      "confidence": 0.92
    }
  ]
}
```

### Fused Detections (Multi-Modal)
```
{
  "timestamp": "2024-01-01T12:00:00Z",
  "fused_detections": [
    {
      "id": 1,
      "class": "car",
      "bbox_2d": [x1, y1, x2, y2],
      "bbox_3d": {"center": [x, y, z], "size": [l, w, h]},
      "velocity": [vx, vy, vz],
      "confidence_camera": 0.95,
      "confidence_lidar": 0.92,
      "confidence_radar": 0.88,
      "fused_confidence": 0.92
    }
  ]
}
```

## 🎮 Examples

Run the example scripts to see the system in action:

```bash
# Camera detection
python examples/example_camera.py

# LiDAR detection
python examples/example_lidar.py

# Radar detection
python examples/example_radar.py

# Multi-modal fusion
python examples/example_fusion.py
```

## 🖼️ Screenshots & Visualizations

### Camera Detection Output
Camera detections are overlaid with bounding boxes and class labels:
- Green boxes: High confidence detections
- Yellow boxes: Medium confidence detections
- Red boxes: Low confidence detections

### LiDAR Detection Output
3D point clouds with annotated bounding boxes:
- Points colored by intensity or height
- 3D boxes around detected objects
- Additional metadata (distance, size)

### Fusion Results
Combined visualization showing:
- 2D camera detections on images
- 3D LiDAR bounding boxes
- Radar velocity vectors
- Unified object IDs across modalities

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model configuration
models:
  camera:
    model_name: "yolov8n"
    confidence_threshold: 0.5
  lidar:
    voxel_size: 0.05
    confidence_threshold: 0.5
  radar:
    range_threshold: 100.0

# Fusion parameters
fusion:
  max_distance_2d_3d_matching: 50  # pixels
  max_distance_3d_matching: 0.5    # meters
  confidence_weights:
    camera: 0.4
    lidar: 0.4
    radar: 0.2
```

## 📈 Performance Metrics

The system tracks:
- Detection accuracy per modality
- Fusion improvement over individual sensors
- Processing time per frame
- False positive/negative rates

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 🤝 Integration with Autonomous Driving Models

The output format is compatible with:
- YOLO autonomous driving models
- Apollo autonomous driving stack
- Carla simulator
- ROS autonomous driving frameworks

## 🔌 Extending the System

### Add a New Detector
1. Create a new detector class inheriting from `BaseDetector`
2. Implement `detect()` and `visualize_results()` methods
3. Register with the fusion engine

### Add a New Model
1. Train or download a model
2. Create a wrapper class for the model
3. Integrate with the appropriate detector

## 📚 Dependencies

- **opencv-python**: Image processing and visualization
- **torch**: Deep learning framework for detection
- **yolov8**: Object detection model
- **open3d**: 3D point cloud processing
- **numpy**: Numerical computations
- **pyyaml**: Configuration file handling
- **scipy**: Scientific computing

## 📝 License

MIT License - Feel free to use and modify for your projects

## 👨‍💻 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Last Updated**: February 2026
**Version**: 1.0.0
