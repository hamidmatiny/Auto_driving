"""
Unit tests for multi-modal object detection system
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_detector import Detection, BaseDetector
from src.camera_detector import CameraDetector
from src.lidar_detector import LiDARDetector
from src.radar_detector import RadarDetector
from src.fusion_engine import FusionEngine


class TestDetection(unittest.TestCase):
    """Test Detection class"""
    
    def test_detection_creation(self):
        """Test creating a detection object"""
        det = Detection(
            class_name="car",
            confidence=0.95,
            bbox=[10, 20, 30, 40]
        )
        self.assertEqual(det.class_name, "car")
        self.assertEqual(det.confidence, 0.95)
        self.assertEqual(det.attributes["bbox"], [10, 20, 30, 40])
    
    def test_detection_to_dict(self):
        """Test converting detection to dictionary"""
        det = Detection(
            class_name="pedestrian",
            confidence=0.85,
            distance=5.0
        )
        det_dict = det.to_dict()
        
        self.assertEqual(det_dict["class"], "pedestrian")
        self.assertEqual(det_dict["confidence"], 0.85)
        self.assertEqual(det_dict["distance"], 5.0)
    
    def test_detection_repr(self):
        """Test detection string representation"""
        det = Detection(class_name="car", confidence=0.9)
        repr_str = repr(det)
        self.assertIn("car", repr_str)
        self.assertIn("0.90", repr_str)


class TestCameraDetector(unittest.TestCase):
    """Test camera detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = CameraDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.model_name, "yolov8n")
        self.assertEqual(self.detector.conf_threshold, 0.5)
    
    def test_filter_by_confidence(self):
        """Test filtering detections by confidence"""
        detections = [
            Detection("car", 0.95),
            Detection("car", 0.6),
            Detection("pedestrian", 0.4)
        ]
        
        filtered = self.detector.filter_by_confidence(detections, 0.7)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].class_name, "car")
        self.assertEqual(filtered[0].confidence, 0.95)
    
    def test_filter_by_class(self):
        """Test filtering detections by class"""
        detections = [
            Detection("car", 0.95),
            Detection("truck", 0.85),
            Detection("pedestrian", 0.8)
        ]
        
        filtered = self.detector.filter_by_class(
            detections, ["car", "truck"]
        )
        self.assertEqual(len(filtered), 2)
        for det in filtered:
            self.assertIn(det.class_name, ["car", "truck"])


class TestLiDARDetector(unittest.TestCase):
    """Test LiDAR detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = LiDARDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.voxel_size, 0.05)
        self.assertEqual(self.detector.min_points, 5)
    
    def test_custom_parameters(self):
        """Test custom detector parameters"""
        detector = LiDARDetector(voxel_size=0.1, min_points=10)
        self.assertEqual(detector.voxel_size, 0.1)
        self.assertEqual(detector.min_points, 10)


class TestRadarDetector(unittest.TestCase):
    """Test radar detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = RadarDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.range_threshold, 100.0)
        self.assertEqual(self.detector.min_doppler, 0.5)
    
    def test_get_moving_objects(self):
        """Test filtering moving objects"""
        detections = [
            Detection("car", 0.95, doppler_velocity=10.0),
            Detection("car", 0.9, doppler_velocity=-5.0),
            Detection("object", 0.8, doppler_velocity=0.0),
            Detection("motorcycle", 0.85, doppler_velocity=15.0)
        ]
        
        # Mock detector methods
        self.detector.detections = detections
        self.detector.min_doppler = 0.5
        
        moving = self.detector.get_moving_objects(detections)
        self.assertEqual(len(moving), 3)
    
    def test_get_stationary_objects(self):
        """Test filtering stationary objects"""
        detections = [
            Detection("car", 0.95, doppler_velocity=10.0),
            Detection("car", 0.9, doppler_velocity=-5.0),
            Detection("object", 0.8, doppler_velocity=0.0),
            Detection("motorcycle", 0.85, doppler_velocity=0.1)
        ]
        
        self.detector.detections = detections
        self.detector.min_doppler = 0.5
        
        stationary = self.detector.get_stationary_objects(detections)
        self.assertEqual(len(stationary), 1)
        self.assertEqual(stationary[0].attributes['doppler_velocity'], 0.0)


class TestFusionEngine(unittest.TestCase):
    """Test sensor fusion engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fusion = FusionEngine()
    
    def test_fusion_initialization(self):
        """Test fusion engine initialization"""
        self.assertIsNotNone(self.fusion.camera_detector)
        self.assertIsNotNone(self.fusion.lidar_detector)
        self.assertIsNotNone(self.fusion.radar_detector)
        
        self.assertEqual(self.fusion.camera_weight, 0.4)
        self.assertEqual(self.fusion.lidar_weight, 0.4)
        self.assertEqual(self.fusion.radar_weight, 0.2)
    
    def test_custom_weights(self):
        """Test custom fusion weights"""
        fusion = FusionEngine(
            camera_weight=0.5,
            lidar_weight=0.3,
            radar_weight=0.2
        )
        
        self.assertEqual(fusion.camera_weight, 0.5)
        self.assertEqual(fusion.lidar_weight, 0.3)
        self.assertEqual(fusion.radar_weight, 0.2)
    
    def test_fuse_class(self):
        """Test class fusion logic"""
        # Test majority voting
        fused_class = self.fusion._fuse_class("car", "car", "truck")
        self.assertEqual(fused_class, "car")
        
        fused_class = self.fusion._fuse_class("car", "truck", "bus")
        self.assertIn(fused_class, ["car", "truck", "bus"])
    
    def test_fuse_confidence(self):
        """Test confidence fusion"""
        sensor_contributions = {
            "camera": {"confidence": 0.95},
            "lidar": {"confidence": 0.90},
            "radar": {"confidence": 0.85}
        }
        
        fused_conf = self.fusion._fuse_confidence(sensor_contributions)
        
        # Should be weighted average
        # 0.95*0.4 + 0.90*0.4 + 0.85*0.2 = 0.38 + 0.36 + 0.17 = 0.91
        expected = 0.95 * 0.4 + 0.90 * 0.4 + 0.85 * 0.2
        
        self.assertAlmostEqual(fused_conf, expected, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_complete_pipeline_with_examples(self):
        """Test complete pipeline with example data"""
        from examples.example_camera import create_sample_image
        from examples.example_lidar import create_sample_lidar_json
        from examples.example_radar import create_sample_radar_data
        
        # Create sample data
        camera_path = create_sample_image()
        lidar_path = create_sample_lidar_json()
        radar_path = create_sample_radar_data()
        
        # Create fusion engine
        fusion = FusionEngine()
        
        # Process
        results = fusion.process(
            camera_data=camera_path,
            lidar_data=lidar_path,
            radar_data=radar_path
        )
        
        # Verify results
        self.assertIn('num_fused_objects', results)
        self.assertIn('fused_detections', results)
        self.assertIn('timestamp', results)
        
        # Check fused detections
        self.assertGreaterEqual(len(results['fused_detections']), 0)
        
        for det in results['fused_detections']:
            self.assertIn('id', det)
            self.assertIn('class', det)
            self.assertIn('confidence', det)
            self.assertIn('sensor_contributions', det)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestCameraDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestLiDARDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestRadarDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("="*70)
    print("Multi-Modal Object Detection - Test Suite")
    print("="*70)
    print()
    
    result = run_tests()
    
    print()
    print("="*70)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
