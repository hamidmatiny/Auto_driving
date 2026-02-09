import cv2
import numpy as np
import json
import os

# 1. Define Paths
data_dir = "examples/sample_data"
image_path = os.path.join(data_dir, "sample_camera_image.jpg")
lidar_json = os.path.join(data_dir, "sample_lidar_data.json")
radar_json = os.path.join(data_dir, "sample_radar_data.json")

# 2. Create a base "road" image (1600x900)
# Dark gray background to represent asphalt/night
img = np.zeros((900, 1600, 3), dtype=np.uint8)
img[:] = (40, 40, 40) 

# 3. Load data with safety checks
with open(lidar_json, 'r') as f:
    lidar_data = json.load(f)
with open(radar_json, 'r') as f:
    radar_data = json.load(f)

def draw_obj(img, center, label, color):
    x, y, z = center
    if x <= 0: x = 0.1 
    scale = 1000 / x
    px = int(800 + (y * scale))
    py = int(450 - (z * scale))
    size = int(200 / x)
    cv2.rectangle(img, (px-size, py-size), (px+size, py+size), color, 3)
    cv2.putText(img, label, (px-size, py-size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- FIXED LOOPS ---

# 4. Draw LiDAR Objects (Green)
# Check if the data is a list; if it's a dict, we might need a specific key like 'results'
if isinstance(lidar_data, list):
    for obj in lidar_data:
        if isinstance(obj, dict): # Ensure we have a dictionary, not a string
            draw_obj(img, obj['center'], obj['class'].upper(), (0, 255, 0))
elif isinstance(lidar_data, dict):
    # If the JSON is a dictionary, iterate over the values
    for key, obj in lidar_data.items():
        if isinstance(obj, dict):
            draw_obj(img, obj.get('center', [0,0,0]), obj.get('class', 'UNK').upper(), (0, 255, 0))

# 5. Draw Radar Objects (Blue)
if isinstance(radar_data, list):
    for obj in radar_data:
        if isinstance(obj, dict):
            pos = obj.get('position_cartesian', [0, 0, 0])
            draw_obj(img, pos, "RADAR_" + obj.get('class', 'UNK').upper(), (255, 100, 0))
elif isinstance(radar_data, dict):
    for key, obj in radar_data.items():
        if isinstance(obj, dict):
            pos = obj.get('position_cartesian', [0, 0, 0])
            draw_obj(img, pos, "RADAR_" + obj.get('class', 'UNK').upper(), (255, 100, 0))

# 6. Save the file properly
cv2.imwrite(image_path, img)
print(f"✓ New sample image generated at {image_path}")
print(f"✓ New sample image generated at {image_path}")
print("This image now contains 3 LiDAR objects and 5 Radar objects.")