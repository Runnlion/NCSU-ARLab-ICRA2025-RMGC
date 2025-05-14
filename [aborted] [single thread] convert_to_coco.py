# Re-imports due to code execution environment reset
import os
import yaml
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Paths
OUTPUT_JSON = "subset1_clear/output_coco_checked.json"
image_root = "subset1_clear/"
gt_files = sorted([f for f in os.listdir(image_root) if f.startswith("GT_left_camera_") and f.endswith(".yaml")])
# COCO structures
images = []
annotations = []
categories = []
category_name_to_id = {}
next_annotation_id = 1

def get_scene_id(filename):
    return int(filename.split('_')[-1].split('.')[0])

# Process one GT file
if gt_files:
    gt_file = image_root + gt_files[0]
    scene_id = get_scene_id(gt_file)
    seg_file = image_root + f"left_camera_scene_{scene_id}_segmentation.png"
    rgb_file = image_root + f"left_camera_point_light_scene_{scene_id}_rgb.png"

    if os.path.exists(seg_file) and os.path.exists(rgb_file):
        seg_rgb = np.array(Image.open(seg_file).convert("RGB"))
        h, w, _ = seg_rgb.shape
        rgb = np.array(Image.open(rgb_file))
        width, height = rgb.shape[1], rgb.shape[0]

        image_id = scene_id
        images.append({
            "id": image_id,
            "file_name": rgb_file,
            "width": width,
            "height": height
        })

        with open(gt_file, 'r') as f:
            gt_data = yaml.safe_load(f)

        for obj_name_with_id, obj_data in gt_data.get('objects', {}).items():
            try:
                class_id_str, class_name = obj_name_with_id.split('-', 1)
                class_id = int(class_id_str)
            except:
                continue

            if class_id not in category_name_to_id.values():
                category_name_to_id[class_name] = class_id
                categories.append({
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "object"
                })

        # Convert RGB color to unique int
        seg_flat = seg_rgb[:, :, 0].astype(np.uint32) * 256**2 + \
                   seg_rgb[:, :, 1].astype(np.uint32) * 256 + \
                   seg_rgb[:, :, 2].astype(np.uint32)
        unique_colors = np.unique(seg_flat)
        color_to_id = {color: idx+1 for idx, color in enumerate(unique_colors) if color != 0}
        seg_id_map = np.zeros((h, w), dtype=np.uint16)
        for color, inst_id in color_to_id.items():
            seg_id_map[seg_flat == color] = inst_id
        seg = seg_id_map

        for inst_id in np.unique(seg):
            if inst_id == 0:
                continue

            matched_name = None
            for obj_name_with_id, obj_data in gt_data.get('objects', {}).items():
                centroid = obj_data.get("2D_centroid", None)
                if centroid is not None:
                    cx, cy = centroid
                    cy_i, cx_i = round(cy), round(cx)
                    if 0 <= cy_i < h and 0 <= cx_i < w:
                        if seg[cy_i, cx_i] == inst_id:
                            matched_name = obj_name_with_id
                            break

            if matched_name is None:
                continue

            class_id_str, class_name = matched_name.split('-', 1)
            category_id = int(class_id_str)
            binary_mask = (seg == inst_id).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            area = mask_utils.area(rle).item()
            bbox = mask_utils.toBbox(rle).tolist()
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = [cnt.flatten().tolist() for cnt in contours if len(cnt.flatten()) >= 6]
            if not segmentation:
                continue
            annotations.append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "bbox_mode": 1,  # ← 1 means BoxMode.XYWH_ABS
                "area": area,
                "iscrowd": 0
            })
            next_annotation_id += 1

        # Visualize
        metadata = MetadataCatalog.get("coco_temp")
        metadata.thing_classes = [cat["name"] for cat in categories]
        v = Visualizer(rgb[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_dict = {
            "file_name": rgb_file,
            "height": height,
            "width": width,
            "annotations": annotations
        }
        
        vis_output = v.draw_dataset_dict(vis_dict)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_output.get_image())
        plt.axis("off")
        plt.title("Preview of instance mask")
        plt.show()

        # Save json
        coco_output = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(coco_output, f)
        print(f"✅ One-image COCO JSON written to {OUTPUT_JSON}")
    else:
        print("❌ Missing segmentation or RGB image.")
else:
    print("❌ No GT yaml files found.")
