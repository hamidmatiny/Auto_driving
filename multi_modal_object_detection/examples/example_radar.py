"""
Example: Radar-based object detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.radar_detector import RadarDetector
import json
import numpy as np


def create_sample_radar_data():
    """Create sample radar detection data"""
    detections = [
        {
            "id": 1,
            "range": 25.5,
            "azimuth": -15.0,
            "elevation": 0.5,
            "doppler_velocity": 12.3,
            "rcs": -5.0,
            "snr": 18.5
        },
        {
            "id": 2,
            "range": 45.2,
            "azimuth": 0.0,
            "elevation": 0.2,
            "doppler_velocity": -8.1,
            "rcs": 8.5,
            "snr": 25.3
        },
        {
            "id": 3,
            "range": 65.8,
            "azimuth": 22.5,
            "elevation": 1.0,
            "doppler_velocity": 0.0,
            "rcs": -15.0,
            "snr": 12.1
        },
        {
            "id": 4,
            "range": 35.0,
            "azimuth": -30.0,
            "elevation": 0.3,
            "doppler_velocity": 18.5,
            "rcs": 5.0,
            "snr": 22.0
        },
        {
            "id": 5,
            "range": 80.0,
            "azimuth": 45.0,
            "elevation": 2.0,
            "doppler_velocity": -25.0,
            "rcs": 0.0,
            "snr": 30.5
        }
    ]
    
    data = {
        "timestamp": "2024-01-01T12:00:00Z",
        "radar_type": "TI IWR6843",
        "num_detections": len(detections),
        "detections": detections
    }
    
    sample_path = "examples/sample_data/sample_radar_data.json"
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    with open(sample_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return sample_path


def main():
    """Run radar detection example"""
    print("\n" + "="*80)
    print("RADAR OBJECT DETECTION EXAMPLE")
    print("="*80 + "\n")
    
    # Create sample data
    print("Creating sample radar data...")
    radar_path = create_sample_radar_data()
    print(f"✓ Sample radar data created at: {radar_path}\n")
    
    # Initialize detector
    print("Initializing Radar Detector...")
    detector = RadarDetector(range_threshold=100.0, min_doppler=0.5)
    print("✓ Detector initialized\n")
    
    # Run detection
    print("Running radar object detection...")
    detections = detector.detect(radar_path)
    print(f"✓ Detection complete: {len(detections)} objects found\n")
    
    # Print results
    detector.visualize_results(detections)
    
    # Separate moving and stationary objects
    moving = detector.get_moving_objects(detections)
    stationary = detector.get_stationary_objects(detections)
    
    print(f"Moving Objects: {len(moving)}")
    print(f"Stationary Objects: {len(stationary)}\n")
    
    if moving:
        print("Moving Objects Details:")
        detector.visualize_results(moving)
    
    if stationary:
        print("Stationary Objects Details:")
        detector.visualize_results(stationary)
    
    # Export results
    print("Exporting detection results...")
    export_data = detector.export_detections(format="json")
    
    output_file = "output/radar_detections.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✓ Results saved to: {output_file}\n")
    
    # Generate range-doppler map
    print("Generating range-doppler map...")
    rdm = detector.generate_range_doppler_map(num_range_bins=256, num_doppler_bins=128)
    print(f"✓ Range-Doppler map shape: {rdm.shape}")
    print(f"  Peak SNR value: {rdm.max():.2f} dB\n")
    
    # Show structure
    print("Detection Structure Example:")
    if detections:
        print(json.dumps(detections[0].to_dict(), indent=2))
    
    print("\n" + "="*80)
    print("Radar Detection Statistics:")
    print("="*80)
    print(f"Total objects detected: {len(detections)}")
    print(f"Mean range: {np.mean([d.attributes['range'] for d in detections]):.1f}m")
    print(f"Range coverage: 0-100m")
    print(f"Velocity range: ±50 m/s")
    print(f"Azimuth coverage: ±180°")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import numpy as np
    main()
