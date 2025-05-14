import os
import cv2
import random
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# Assign the path to your JSON file and image root directory
json_file = "coco_with_multiple_lighting_multithreaded_updated.json"
image_root = ""  # Image root directory

# Register the dataset with Detectron2
dataset_name = "icra_rmgc_original"
register_coco_instances(dataset_name, {}, json_file, image_root)

# Fetch the dataset and metadata
dataset_dicts = DatasetCatalog.get(dataset_name)
metadata = MetadataCatalog.get(dataset_name)

# Randomly shuffle the dataset to visualize a random sample
random.shuffle(dataset_dicts)
sample = None
for d in dataset_dicts:
    if len(d["annotations"]) > 0:
        sample = d
        break

if sample is None:
    print("❌ No annotations found in the dataset.")
    exit()

# Load RGB image
img_path = os.path.join(image_root, sample["file_name"])
image = cv2.imread(img_path)
if image is None:
    print(f"❌ Image not found: {img_path}")
    exit()

# Visualize the image with annotations
visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
out = visualizer.draw_dataset_dict(sample)

# Display the image with annotations
plt.figure(figsize=(12, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.title(sample["file_name"])
plt.show()
