import json
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from collections import defaultdict

# Mapping from Roboflow internal category IDs to logical IDs
category_id_remap = {
    1: 1, 2: 10, 3: 2, 4: 3, 5: 31, 6: 32, 7: 33, 8: 34, 9: 35, 10: 36,
    11: 37, 12: 38, 13: 39, 14: 4, 15: 40, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9
}

def polygons_to_shapely(segmentation):
    polys = []
    for seg in segmentation:
        pts = np.array(seg).reshape(-1, 2)
        polys.append(Polygon(pts))
    return unary_union(polys)

def merge_group(group, new_id):
    merged_poly = []
    for ann in group:
        merged_poly.extend(ann["segmentation"])
    union = polygons_to_shapely(merged_poly)
    bbox = list(union.bounds)
    area = union.area
    if union.geom_type == 'MultiPolygon':
        segmentation = [list(np.array(p.exterior.coords).flatten()) for p in union.geoms]
    else:
        segmentation = [list(np.array(union.exterior.coords).flatten())]
    return {
        "id": new_id,
        "image_id": group[0]["image_id"],
        "category_id": category_id_remap.get(group[0]["category_id"], group[0]["category_id"]),
        "segmentation": segmentation,
        "iscrowd": 0,
        "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
        "area": area,
    }

def coco_merge_instances(coco):
    new_annotations = []
    next_id = max(ann["id"] for ann in coco["annotations"]) + 1
    image_ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        image_ann_map[ann["image_id"]].append(ann)

    for anns in image_ann_map.values():
        grouped, visited = [], set()
        for i, ann1 in enumerate(anns):
            if i in visited:
                continue
            group, poly1 = [ann1], polygons_to_shapely(ann1["segmentation"])
            for j, ann2 in enumerate(anns):
                if j <= i or j in visited: continue
                if category_id_remap.get(ann1["category_id"]) == category_id_remap.get(ann2["category_id"]):
                    poly2 = polygons_to_shapely(ann2["segmentation"])
                    if poly1.intersects(poly2) or poly1.distance(poly2) < 500:
                        group.append(ann2)
                        visited.add(j)
            visited.add(i)
            grouped.append(group)

        for group in grouped:
            if len(group) == 1:
                ann = group[0]
                ann["category_id"] = category_id_remap.get(ann["category_id"], ann["category_id"])
                new_annotations.append(ann)
            else:
                merged = merge_group(group, next_id)
                new_annotations.append(merged)
                next_id += 1

    coco["annotations"] = new_annotations
    return coco

def update_categories(original_categories):
    new_categories = []
    logical_ids_used = set()
    for cat in original_categories:
        internal_id = cat["id"]
        name = cat["name"]
        logical_id = category_id_remap.get(internal_id)
        if logical_id is not None and logical_id not in logical_ids_used:
            new_categories.append({
                "id": logical_id,
                "name": name,  # keep "1-Cheez-it"
                "supercategory": cat.get("supercategory", "none")
            })
            logical_ids_used.add(logical_id)
    return sorted(new_categories, key=lambda x: x["id"])

def process_and_save(input_json, output_json):
    with open(input_json) as f:
        coco = json.load(f)
    coco = coco_merge_instances(coco)
    coco["categories"] = update_categories(coco["categories"])
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

# Example usage:
# process_and_save("roboflow_export.json", "final_merged_annotations.coco.json")

process_and_save('/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/NCSU-ARLab-ICRA2025-RMGC/dataset/train/_annotations.coco.json', "converted.json")