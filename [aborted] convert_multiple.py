import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as mask_utils

# Output Files
OUTPUT_JSON = "coco_with_multiple_lighting_updated.json"

# Color Mapping: RGB Encoding → (name, category_id)
color_map = {
    13473036: ("Pringles", 40),
    2321294: ("Glove", 39),
    13303664: ("Rolodex_Jumbo_Pencil_Cup", 38),
    15627809: ("T-shirt", 37),
    16728128: ("Adjustable_Wrench", 36),
    9127187: ("Laugh_Out_Loud_Joke_Book", 35),
    39629: ("Nail", 34),
    3100495: ("Key", 33),
    9315107: ("Wine_Glass", 32),
    9139456: ("Plastic_White_Cup", 31),
    255: ("meat_can", 10),
    65280: ("Mug", 9),
    16711680: ("Plastic_Banana", 8),
    65407: ("Clamp", 7),
    8323327: ("Foam_Brick", 6),
    16744192: ("Tomato_Soup", 5),
    8388607: ("Frenchs_Mustard", 4),
    16744447: ("Scissors", 3),
    16777087: ("Starkist_Tuna", 2),
    11711154: ("Cheez-it", 1)
}

# Supported Lighting Conditions
light_conditions = ["point_light", "spot_light", "directional_light"]

# COCO Categories
images, annotations, categories = [], [], []
next_ann_id = 1

# COCO Categories
for color_id, (name, cid) in color_map.items():
    categories.append({
        "id": cid,
        "name": name,
        "supercategory": "object"
    })
root = "/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/icra_comp/"

# Find segmentation files
seg_1 = sorted([f for f in os.listdir('./subset1_clear/') if f.startswith("left_camera_scene_") and f.endswith("_segmentation.png")])
seg_4 = sorted([f for f in os.listdir('./subset4_clear/') if f.startswith("left_camera_scene_") and f.endswith("_segmentation.png")])
assert seg_1 == seg_4
seg_files = seg_1
for folder in ['./subset1_clear/', './subset4_clear/']:
    if folder == './subset1_clear/':
        subset = 'subset1'
    else:
        subset = 'subset4'

    for seg_file in tqdm(seg_files, desc=f"Processing segmentation masks, Folder: {folder}"):
        scene_id = int(seg_file.split('_')[3])  # 提取 ID
        seg_rgb = np.array(Image.open(f"{folder}{seg_file}").convert("RGB"))
        h, w, _ = seg_rgb.shape

        seg_flat = seg_rgb[:, :, 0].astype(np.uint32) * 256**2 + \
                seg_rgb[:, :, 1].astype(np.uint32) * 256 + \
                seg_rgb[:, :, 2].astype(np.uint32)

        for light in light_conditions:
            rgb_file = f"{folder}left_camera_{light}_scene_{scene_id}_rgb.png"
            if not os.path.exists(rgb_file):
                print(f"❌ Missing RGB for {light}, scene {scene_id}")
                continue
            rgb = Image.open(rgb_file)
            width, height = rgb.size

            image_id = f"{subset}_{scene_id}_{light}"
            images.append({
                "id": image_id,
                "file_name": rgb_file,
                "width": width,
                "height": height
            })

            # Traverse unique colors in the segmentation mask
            for color_id in np.unique(seg_flat):
                if color_id == 0 or color_id not in color_map:
                    continue

                class_name, category_id = color_map[color_id]
                binary_mask = (seg_flat == color_id).astype(np.uint8)

                if np.sum(binary_mask) == 0:
                    continue

                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                area = mask_utils.area(rle).item()
                bbox = mask_utils.toBbox(rle).tolist()

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = [cnt.flatten().tolist() for cnt in contours if len(cnt.flatten()) >= 6]
                if not segmentation:
                    continue

                annotations.append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "color_id": int(color_id)
                })
                next_ann_id += 1

        # if(next_ann_id >= 50): 
        #     next_ann_id = 0
        #     break

# Write to JSON
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_output, f, indent=2)

