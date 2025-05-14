import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
from pycocotools import mask as mask_utils

# Output Files
OUTPUT_JSON = "coco_with_multiple_lighting_multithreaded_updated.json"

# Color Mapping: RGB Encoding → (name, category_id)
color_map = {
    13473036: ("40-Pringles", 40),
    2321294: ("39-Glove", 39),
    13303664: ("38-Rolodex_Jumbo_Pencil_Cup", 38),
    15627809: ("37-T-shirt", 37),
    16728128: ("36-Adjustable_Wrench", 36),
    9127187: ("35-Laugh_Out_Loud_Joke_Book", 35),
    39629: ("34-Nail", 34),
    3100495: ("33-Key", 33),
    9315107: ("32-Wine_Glass", 32),
    9139456: ("31-Plastic_White_Cup", 31),
    255: ("10-meat_can", 10),
    65280: ("9-Mug", 9),
    16711680: ("8-Plastic_Banana", 8),
    65407: ("7-Clamp", 7),
    8323327: ("6-Foam_Brick", 6),
    16744192: ("5-Tomato_Soup", 5),
    8388607: ("4-Frenchs_Mustard", 4),
    16744447: ("3-Scissors", 3),
    16777087: ("2-Starkist_Tuna", 2),
    11711154: ("1-Cheez-it", 1)
}

# Supported Lighting Conditions
light_conditions = ["point_light", "spot_light", "directional_light"]

# COCO Categories
categories = [{
    "id": cid,
    "name": name,
    "supercategory": "object"
} for _, (name, cid) in color_map.items()]

# Function to process each scene
def process_scene(args):
    folder, subset, seg_file, next_ann_id_base = args
    local_images, local_annotations = [], []
    scene_id = int(seg_file.split('_')[3])
    seg_path = os.path.join(folder, seg_file)
    seg_rgb = np.array(Image.open(seg_path).convert("RGB"))
    h, w, _ = seg_rgb.shape
    seg_flat = seg_rgb[:, :, 0].astype(np.uint32) * 256**2 + \
               seg_rgb[:, :, 1].astype(np.uint32) * 256 + \
               seg_rgb[:, :, 2].astype(np.uint32)

    local_ann_id = next_ann_id_base
    for light in light_conditions:
        rgb_file = f"left_camera_{light}_scene_{scene_id}_rgb.png"
        rgb_path = os.path.join(folder, rgb_file)
        if not os.path.exists(rgb_path):
            continue
        rgb = Image.open(rgb_path)
        width, height = rgb.size
        image_id = f"{subset}_{scene_id}_{light}"
        local_images.append({
            "id": image_id,
            "file_name": rgb_path,
            "width": width,
            "height": height
        })

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

            local_annotations.append({
                "id": local_ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "color_id": int(color_id)
            })
            local_ann_id += 1
    return local_images, local_annotations

# Main Processing
tasks = []
folders = [('./subset1_clear/', 'subset1'), ('./subset4_clear/', 'subset4')]
seg_files = sorted([f for f in os.listdir('./subset1_clear/') if f.startswith("left_camera_scene_") and f.endswith("_segmentation.png")])
ann_id_counter = 1
for folder_path, subset in folders:
    for seg_file in seg_files:
        tasks.append((folder_path, subset, seg_file, ann_id_counter))
        ann_id_counter += 1000

# Conduct multiprocessing for accelereation
with Pool(processes=32) as pool:
    results = list(tqdm(pool.imap_unordered(process_scene, tasks), total=len(tasks)))

# Sort and combine results
images, annotations = [], []
for imgs, anns in results:
    images.extend(imgs)
    annotations.extend(anns)

# Update image IDs and  annotation_id
for new_id, ann in enumerate(annotations, start=1):
    ann["id"] = new_id

# Write COCO JSON
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_output, f, indent=2)