"""
Example: Multi-modal sensor fusion
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion_engine import FusionEngine
from examples.example_camera import create_sample_image
from examples.example_lidar import create_sample_lidar_json
from examples.example_radar import create_sample_radar_data
import json


def main():
    """Run complete multi-modal fusion example"""
    print("\n" + "="*80)
    print("MULTI-MODAL SENSOR FUSION EXAMPLE")
    print("="*80 + "\n")
    
    # Create sample data for all sensors
    print("Creating sample sensor data...")
    print("  - Camera image...", end=" ")
    camera_path = create_sample_image()
    print("✓")
    
    print("  - LiDAR point cloud...", end=" ")
    lidar_path = create_sample_lidar_json()
    print("✓")
    
    print("  - Radar measurements...", end=" ")
    radar_path = create_sample_radar_data()
    print("✓\n")
    
    # Initialize fusion engine
    print("Initializing Sensor Fusion Engine...")
    fusion = FusionEngine(
        camera_weight=0.4,
        lidar_weight=0.4,
        radar_weight=0.2
    )
    print("✓ Fusion engine initialized\n")
    
    # Process multi-modal data
    print("Processing multi-modal sensor data...")
    print("-" * 80)
    results = fusion.process(
        camera_data=camera_path,
        lidar_data=lidar_path,
        radar_data=radar_path
    )
    print("-" * 80 + "\n")
    
    # Visualize fusion results
    print("Fusion Results Summary:")
    fusion.visualize_fusion_results(results)
    
    # Export fused detections
    print("Exporting fused detection results...")
    output_file = "output/fused_detections.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Fused results saved to: {output_file}\n")
    
    # Show detailed fusion example
    print("="*80)
    print("Detailed Fusion Results Example:")
    print("="*80 + "\n")
    
    if results['fused_detections']:
        # Show first fused detection in detail
        det = results['fused_detections'][0]
        print(f"Object ID: {det['id']}")
        print(f"Class: {det['class']}")
        print(f"Position (3D): ({det['position_3d'][0]:.2f}, {det['position_3d'][1]:.2f}, {det['position_3d'][2]:.2f})")
        print(f"Fused Confidence: {det['confidence']:.2%}\n")
        
        print("Sensor Contributions:")
        for sensor, contrib in det['sensor_contributions'].items():
            print(f"\n  {sensor.upper()}:")
            for key, value in contrib.items():
                if isinstance(value, float):
                    print(f"    - {key}: {value:.4f}")
                else:
                    print(f"    - {key}: {value}")
    
    # Statistics
    print("\n" + "="*80)
    print("FUSION STATISTICS")
    print("="*80)
    print(f"\nIndividual Sensor Detections:")
    print(f"  - Camera:  {results['sensor_detections']['camera']} objects")
    print(f"  - LiDAR:   {results['sensor_detections']['lidar']} objects")
    print(f"  - Radar:   {results['sensor_detections']['radar']} objects")
    print(f"  - Total:   {sum(results['sensor_detections'].values())} detections")
    
    print(f"\nFused Results:")
    print(f"  - Total fused objects: {results['num_fused_objects']}")
    
    # Analyze sensor contribution
    sensor_usage = {"camera": 0, "lidar": 0, "radar": 0}
    for det in results['fused_detections']:
        for sensor in det['sensor_contributions'].keys():
            sensor_usage[sensor] += 1
    
    print(f"\nSensor Contribution to Fused Detections:")
    print(f"  - Camera: {sensor_usage['camera']} / {results['num_fused_objects']} ({100*sensor_usage['camera']/max(1, results['num_fused_objects']):.1f}%)")
    print(f"  - LiDAR:  {sensor_usage['lidar']} / {results['num_fused_objects']} ({100*sensor_usage['lidar']/max(1, results['num_fused_objects']):.1f}%)")
    print(f"  - Radar:  {sensor_usage['radar']} / {results['num_fused_objects']} ({100*sensor_usage['radar']/max(1, results['num_fused_objects']):.1f}%)")
    
    # Confidence statistics
    if results['fused_detections']:
        confidences = [d['confidence'] for d in results['fused_detections']]
        print(f"\nConfidence Statistics:")
        print(f"  - Mean confidence: {sum(confidences)/len(confidences):.2%}")
        print(f"  - Min confidence:  {min(confidences):.2%}")
        print(f"  - Max confidence:  {max(confidences):.2%}")
    
    print("\n" + "="*80)
    print("Multi-Modal Fusion Example Completed!")
    print("="*80 + "\n")
    
    # Output file locations
    print("Generated Output Files:")
    print(f"  📄 Fused detections:    {output_file}")
    print(f"  📊 Camera detections:   output/camera_detections.json")
    print(f"  📊 LiDAR detections:    output/lidar_detections.json")
    print(f"  📊 Radar detections:    output/radar_detections.json")
    print()


if __name__ == "__main__":
    main()
