"""
Example: Camera-based object detection using YOLOv8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera_detector import CameraDetector
import json
import numpy as np
from PIL import Image


def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple test image (640x480)
    img = Image.new('RGB', (640, 480), color='white')
    
    # Add some shapes to simulate a scene
    pixels = img.load()
    
    # Draw a rectangle (car simulation)
    for x in range(100, 300):
        for y in range(150, 300):
            pixels[x, y] = (100, 100, 100)
    
    # Draw a person-like shape
    for x in range(400, 450):
        for y in range(50, 200):
            pixels[x, y] = (200, 200, 200)
    
    sample_path = "examples/sample_data/sample_camera_image.jpg"
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    img.save(sample_path)
    return sample_path


def main():
    """Run camera detection example"""
    print("\n" + "="*80)
    print("CAMERA DETECTION EXAMPLE")
    print("="*80 + "\n")
    
    # Create sample image
    print("Creating sample image...")
    image_path = "examples/sample_data/sample_image.jpg"
    print(f"✓ Sample image created at: {image_path}\n")
    
    # Initialize detector
    print("Initializing Camera Detector...")
    detector = CameraDetector(model_name="yolov8n", conf_threshold=0.5)
    print("✓ Detector initialized\n")
    
    # Run detection
    print("Running object detection...")
    detections = detector.detect(image_path)
    print(f"✓ Detection complete: {len(detections)} objects found\n")
    
    # Print results
    detector.print_detections(detections)
    
    # Export results
    print("Exporting detection results...")
    export_data = detector.export_detections(format="json")
    
    output_file = "output/camera_detections.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✓ Results saved to: {output_file}\n")
    
    # Show structure of detections
    print("Detection Structure Example:")
    if detections:
        print(json.dumps(detections[0].to_dict(), indent=2))
    
    # Visualization
    print("\nVisualizing results...")
    try:
        detector.visualize_results(
            detections=detections,
            save_path="output/annotated_camera_image.jpg"
        )
        print("✓ Visualization complete")
    except Exception as e:
        print(f"Note: Visualization requires display. Error: {e}")
        print("But results have been saved to: output/annotated_camera_image.jpg")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
