import json
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def coco_merge_instances(coco_path, output_path):
    with open(coco_path) as f:
        coco = json.load(f)

    new_annotations = []
    used = set()
    next_id = max(ann["id"] for ann in coco["annotations"]) + 1

    # Group annotations by image
    from collections import defaultdict
    image_ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        image_ann_map[ann["image_id"]].append(ann)

    for image_id, anns in image_ann_map.items():
        grouped = []
        visited = set()

        for i, ann1 in enumerate(anns):
            if i in visited:
                continue
            group = [ann1]
            poly1 = polygons_to_shapely(ann1["segmentation"])

            for j, ann2 in enumerate(anns):
                if j <= i or j in visited:
                    continue
                if ann1["category_id"] == ann2["category_id"]:
                    poly2 = polygons_to_shapely(ann2["segmentation"])
                    if poly1.intersects(poly2) or poly1.distance(poly2) < 10:
                        group.append(ann2)
                        visited.add(j)
            visited.add(i)
            grouped.append(group)

        for group in grouped:
            if len(group) == 1:
                new_annotations.append(group[0])
            else:
                merged = merge_group(group, next_id)
                new_annotations.append(merged)
                next_id += 1

    coco["annotations"] = new_annotations

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

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

    merged_ann = {
        "id": new_id,
        "image_id": group[0]["image_id"],
        "category_id": group[0]["category_id"],
        "segmentation": [list(np.array(p.exterior.coords).flatten()) for p in union.geoms] if union.geom_type == 'MultiPolygon' else [list(np.array(union.exterior.coords).flatten())],
        "iscrowd": 0,
        "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
        "area": area,
    }
    return merged_ann

coco_merge_instances('/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/NCSU-ARLab-ICRA2025-RMGC/dataset/train/_annotations.coco.json', "converted.json")