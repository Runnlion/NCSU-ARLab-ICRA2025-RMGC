import os
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === Input Paths (modify as needed) ===
subset_name = "subset4_clear" # Change to your subset name, e.g., "subset1_clear" or "subset2_clear"
image_id = '102'
yaml_file = f"./{subset_name}/GT_left_camera_{image_id}.yaml"
rgb_image_path = f"./{subset_name}/left_camera_directional_light_scene_{image_id}_rgb.png"
seg_image_path = f"./{subset_name}/left_camera_scene_{image_id}_segmentation.png"

# === Load YAML ===
with open(yaml_file, 'r') as f:
    gt_data = yaml.safe_load(f)

class_name_map = {}
for obj_name_with_id in gt_data.get("objects", {}):
    try:
        class_id_str, class_name = obj_name_with_id.split('-', 1)
        class_id = int(class_id_str)
        class_name_map[class_name] = class_id
    except:
        continue

# === Load Images ===
rgb = cv2.cvtColor(cv2.imread(rgb_image_path), cv2.COLOR_BGR2RGB)
seg = np.array(Image.open(seg_image_path).convert("RGB"))

# === Build color-to-class mapping (if needed) ===
seg_flat = seg[:, :, 0].astype(np.uint32) * 256**2 + \
           seg[:, :, 1].astype(np.uint32) * 256 + \
           seg[:, :, 2].astype(np.uint32)
unique_colors = np.unique(seg_flat)

print("📋 Unique instance colors in segmentation image:", len(unique_colors))
print("📂 Class names from YAML:")
for cname in class_name_map:
    print(f" - ID: {class_name_map[cname]}, {cname}")

# === UI Callback ===
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = seg[y, x]
        rgb_val = tuple(bgr.tolist())
        flat_val = rgb_val[0] * 256**2 + rgb_val[1] * 256 + rgb_val[2]
        print(f"🖱️ Click at ({x}, {y}) — Segmentation RGB: {rgb_val} | ID: {flat_val}")

# === Display with zoom ===
window_name = "RGB Image (click to inspect pixel)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, on_mouse)

print("\n🔍 Click on the image to inspect pixel color (segmentation ID). Press [q] to quit.")

while True:
    cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
