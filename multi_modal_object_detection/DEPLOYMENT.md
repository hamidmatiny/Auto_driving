# DEPLOYMENT & SYSTEM REQUIREMENTS

## System Requirements

### Minimum Requirements (CPU Only)

```
OS: macOS 10.14+, Linux (Ubuntu 18.04+), Windows 10+
Python: 3.8 - 3.11
RAM: 4 GB minimum, 8 GB recommended
Storage: 2 GB for dependencies
Processor: Intel i5 / AMD Ryzen 5 or equivalent
```

### Recommended Configuration (GPU Acceleration)

```
OS: Ubuntu 20.04 LTS or macOS 12+
Python: 3.9 - 3.11
RAM: 16 GB
Storage: 10 GB SSD
GPU: NVIDIA (CUDA 11.8+) or Apple Silicon
Processor: Intel i7 / AMD Ryzen 7 or equivalent
```

### Performance Tiers

#### Entry Level (CPU)
- Processing: 5 FPS (fusion)
- Latency: 200ms
- Cost: Low (can use laptop)
- Use: Development, prototyping

#### Standard (GPU)
- Processing: 25 FPS (fusion)
- Latency: 40ms
- Cost: Medium ($500-1000 GPU)
- Use: Research, testing

#### Production (High-End GPU)
- Processing: 50+ FPS (fusion)
- Latency: 20ms
- Cost: High ($2000+ GPU)
- Use: Real-time autonomous driving

---

## Installation Guide

### macOS Setup

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Install homebrew packages (if needed)
brew install python3 cmake

# 3. Navigate to project
cd ..

# 4. Create virtual environment
python3 -m venv venv

# 5. Activate environment
source venv/bin/activate

# 6. Upgrade pip
pip install --upgrade pip setuptools wheel

# 7. Install dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import torch; print(torch.__version__)"
```

### Linux Setup (Ubuntu 20.04)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install python3.9 python3-pip python3-venv
sudo apt-get install cmake libopenblas-dev liblapack-dev

# 2. Navigate to project
cd /path/to/multi_modal_object_detection

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate environment
source venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch (with CUDA support if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7. Install other dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### Windows Setup

```cmd
# 1. Check Python
python --version

# 2. Navigate to project
cd path\to\multi_modal_object_detection

# 3. Create virtual environment
python -m venv venv

# 4. Activate environment
venv\Scripts\activate

# 5. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 6. Install PyTorch (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7. Install other dependencies
pip install -r requirements.txt

# 8. Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## GPU Setup

### NVIDIA CUDA Setup (Linux)

```bash
# 1. Download CUDA Toolkit 11.8
# Visit: https://developer.nvidia.com/cuda-11-8-0-download-archive

# 2. Install CUDA
sudo sh cuda_11.8.0_515.43.04_linux.run

# 3. Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Verify CUDA
nvidia-smi
nvcc --version

# 5. Install cuDNN
# Download from: https://developer.nvidia.com/cudnn
# Follow installation instructions

# 6. Test with PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Apple Silicon (M1/M2/M3) Setup

```bash
# 1. Create environment for Apple Silicon
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch for Apple Silicon
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Verify MPS (Metal Performance Shaders)
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "examples/example_fusion.py"]
```

### Build and Run Docker

```bash
# Build image
docker build -t multi-modal-detection:latest .

# Run container
docker run --rm multi-modal-detection:latest

# With GPU support
docker run --rm --gpus all multi-modal-detection:latest

# With volume mount
docker run --rm -v /path/to/data:/app/data multi-modal-detection:latest
```

---

## Cloud Deployment

### AWS EC2 Setup

```bash
# 1. Launch EC2 instance
# - AMI: Deep Learning AMI (Ubuntu 20.04)
# - Instance: p3.2xlarge (with GPU) or t3.xlarge (CPU only)

# 2. Connect via SSH
ssh -i your_key.pem ubuntu@your_instance_ip

# 3. Install dependencies
sudo apt-get update
pip install -r requirements.txt

# 4. Upload data
scp -i your_key.pem -r data/ ubuntu@your_instance_ip:/home/ubuntu/

# 5. Run processing
python examples/example_fusion.py

# 6. Download results
scp -i your_key.pem -r ubuntu@your_instance_ip:/home/ubuntu/output ./
```

### Google Colab Setup

```python
# 1. Upload to Colab
from google.colab import files
uploaded = files.upload()

# 2. Install dependencies
!pip install ultralytics opencv-python open3d torch torchvision

# 3. Run detection
!python examples/example_fusion.py

# 4. Download results
files.download('output/fused_detections.json')
```

---

## Monitoring & Logging

### Enable Debug Logging

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug("Detection started")
logger.info("Found 5 objects")
logger.warning("Low confidence detection")
logger.error("Failed to load image")
```

### Performance Monitoring

```python
import time
from src.fusion_engine import FusionEngine

def monitor_performance():
    fusion = FusionEngine()
    
    # Measure processing time
    start = time.time()
    results = fusion.process(camera_data, lidar_data, radar_data)
    elapsed = time.time() - start
    
    fps = 1.0 / elapsed
    print(f"Processing time: {elapsed*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory:.2f}MB")
```

---

## Optimization Tips

### For CPU-Only Systems

```python
# Use smaller models
detector = CameraDetector(model_name="yolov8n")  # nano

# Reduce image size
config:
  camera:
    imgsz: 416  # instead of 640

# Process fewer frames per second
import time
for frame in stream:
    results = detector.detect(frame)
    time.sleep(0.1)  # 10 FPS instead of 30
```

### For GPU Systems

```yaml
# config.yaml
models:
  camera:
    device: "cuda"
    fp16: true              # Half precision for speed
    optimize_model: true    # TorchScript optimization

processing:
  batch_processing: true
  num_workers: 4           # Parallel processing
```

### Memory Optimization

```python
# Clear unnecessary data
import gc

results = fusion.process(...)
# Process results...
del fusion
del results
gc.collect()  # Force garbage collection
```

---

## Production Deployment Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] All tests passing (`python tests/test_detectors.py`)
- [ ] Example scripts run successfully
- [ ] GPU configured (if using GPU)
- [ ] Configuration file tuned for your system
- [ ] Input data format validated
- [ ] Output paths created
- [ ] Logging configured
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Documentation reviewed
- [ ] Version control initialized
- [ ] Backup strategy in place

---

## Troubleshooting Deployment

### Issue: "torch not found"

```bash
# Solution
pip install torch torchvision torchaudio
```

### Issue: "CUDA out of memory"

```python
# Solution: Reduce batch size or use CPU
config:
  models:
    camera:
      device: "cpu"
      imgsz: 416  # smaller size
```

### Issue: "Open3D import error"

```bash
# Solution
pip install open3d --upgrade
```

### Issue: "Slow inference on CPU"

```python
# Solution: Use quantized model
from ultralytics import YOLO

model = YOLO('yolov8n-int8.pt')  # Quantized model
```

---

## Maintenance

### Regular Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### System Health Check

```python
def health_check():
    import torch
    import cv2
    import open3d as o3d
    
    print("Python dependencies:")
    print(f"  ✓ torch: {torch.__version__}")
    print(f"  ✓ cv2: {cv2.__version__}")
    print(f"  ✓ open3d: {o3d.__version__}")
    
    print("\nGPU Status:")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")

health_check()
```

---

## Performance Benchmarking

### Benchmark Script

```python
import time
import numpy as np
from src.fusion_engine import FusionEngine

def benchmark():
    fusion = FusionEngine()
    
    num_runs = 10
    times = []
    
    for i in range(num_runs):
        start = time.time()
        results = fusion.process(
            camera_data="sample.jpg",
            lidar_data="sample.pcd",
            radar_data="sample.json"
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\nBenchmark Results:")
    print(f"  Mean: {np.mean(times)*1000:.2f}ms")
    print(f"  Std: {np.std(times)*1000:.2f}ms")
    print(f"  Min: {np.min(times)*1000:.2f}ms")
    print(f"  Max: {np.max(times)*1000:.2f}ms")
    print(f"  FPS: {1/np.mean(times):.2f}")

benchmark()
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          
      - name: Run tests
        run: |
          python tests/test_detectors.py
```

---

## Support & Resources

- **Documentation**: See `/docs` folder
- **Examples**: See `/examples` folder
- **Tests**: See `/tests` folder
- **Issues**: GitHub Issues page
- **Community**: GitHub Discussions

---

**Version**: 1.0.0  
**Last Updated**: February 2026
