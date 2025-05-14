import cv2
import random
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# === Step 1: Load Config ===
cfg = get_cfg()
cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # your category count
# cfg.MODEL.WEIGHTS = os.path.join("output_icra_segment", "model_final.pth")  # trained model
cfg.MODEL.WEIGHTS = os.path.join("./output_icra_segment/mutliple_train_3", "model_0008999.pth")  # trained model
cfg.DATASETS.TEST = ("icra_train",)


# Setting thing_classes for visualization
# MetadataCatalog.get("icra_train").thing_classes = [
#     "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard", "5-Tomato_Soup",
#     "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana", "9-Mug", "10-meat_can",
#     "31-Plastic_White_Cup", "32-Wine_Glass", "33-Key", "34-Nail", "35-Laugh_Out_Loud_Joke_Book",
#     "36-Adjustable_Wrench", "37-T-shirt", "38-Rolodex_Jumbo_Pencil_Cup", "39-Glove", "40-Pringles"
# ]

metadata = MetadataCatalog.get("icra_train")
predictor = DefaultPredictor(cfg)

# === Step 2: Load test image ===

# image_path = "/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/icra_comp/left_camera_point_light_scene_1_rgb.png"  # Replace with actual image
# image_path = "/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/icra_comp/left_camera_point_light_scene_156_rgb.png"  # Replace with actual image
image_path = "/home/wolftech/lxiang3.lab/Desktop/sdu6_inside/icra_comp/19_image.jpg"  # Replace with actual image
image = cv2.imread(image_path)

# === Step 3: Inference ===
outputs = predictor(image)
print(outputs)
# === Step 4: Visualization ===
v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save or show
cv2.imshow("Prediction", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()