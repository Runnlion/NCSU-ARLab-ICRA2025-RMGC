import os
import detectron2
import uuid
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# === Set the path ===
json_root = "./"
image_root = "./"
# json_file = json_root + "test_with_color_map.json"
# json_file = json_root + "coco_with_multiple_lighting.json"
json_file = json_root + "coco_with_multiple_lighting_multithreaded_updated.json"

# === Data Registering ===
register_coco_instances("icra_train", {}, json_file, image_root)

# Check if the dataset is registered correctly
metadata = MetadataCatalog.get("icra_train")
print("Registered categories:", metadata)


# === Configure the model ===
cfg = get_cfg()
# Use relative path to the config file
cfg.merge_from_file(f"./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("icra_train",)
cfg.DATASETS.TEST = ()  # No Validation dataset at the moment
cfg.DATALOADER.NUM_WORKERS = 20

cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025  # leraning rate
cfg.SOLVER.MAX_ITER = 15000     # maximum iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES =  20  # classes in the dataset, in this case, 20
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # Save checkpoint every 1000 iterations


cfg.OUTPUT_DIR = f"./output_icra_segment/mutliple_train_{uuid.uuid4()}" # Unique output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === 🚀 Start Training ===
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
