import os
import detectron2
import uuid
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# === Set the path ===
json_root = "./"
image_root_train = "./dataset/train/"
image_root_valid = "./dataset/valid/"
folder = "./output_icra_segment/mutliple_train_3"

json_file_train = "./dataset/train/converted.json"
json_file_valid = "./dataset/valid/converted.json"

# === Data Registering ===
register_coco_instances("realworld_finetune_train", {}, json_file_train, image_root_train)
register_coco_instances("realworld_finetune_val", {}, json_file_valid, image_root_valid)

# === Configure the model ===
cfg = get_cfg()
# Use relative path to the config file
cfg.merge_from_file(f"./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("realworld_finetune_train",)
cfg.DATASETS.TEST = ("realworld_finetune_val",)
cfg.DATALOADER.NUM_WORKERS = 20


cfg.MODEL.WEIGHTS = f"{folder}/model_final.pth"  # Path to the pre-trained model
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00005  # leraning rate
cfg.SOLVER.MAX_ITER = 2000     # maximum iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES =  20  # classes in the dataset, in this case, 20
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # Save checkpoint every 1000 iterations
cfg.TEST.EVAL_PERIOD = 500 


cfg.OUTPUT_DIR = f"{folder}/finetune_{uuid.uuid4()}" # Unique output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# === Trainer with COCO Evaluator support ===
class CocoTrainerWithVal(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
# === 🚀 Step 3: Fine-tune ===
# trainer = DefaultTrainer(cfg)
trainer = CocoTrainerWithVal(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# === ✅ Optional: Evaluate after training ===
evaluator = COCOEvaluator("realworld_finetune_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_final"))
val_loader = build_detection_test_loader(cfg, "realworld_finetune_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

