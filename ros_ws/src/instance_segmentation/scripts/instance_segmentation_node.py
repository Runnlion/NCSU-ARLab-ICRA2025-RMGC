#!/bin python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import rospkg, os
from icra_msgs.msg import DetectionResult, DetectionInstance  # You will need to define these message types
class InstanceSegmentationNode:
    def __init__(self):
        rospy.init_node("instance_segmentation_node", anonymous=True)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('instance_segmentation')
        base_path = os.path.abspath(os.path.join(package_path, '../../../'))  # two levels up


        cfg = get_cfg()
        cfg.merge_from_file(f"{base_path}/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        cfg.MODEL.WEIGHTS = f"{base_path}/output_icra_segment/mutliple_train_3/finetune_bf4ee699-0f29-4a39-a7d6-025fafbc4c66/model_final.pth"
        cfg.DATASETS.TEST = ("icra_realtime",)

        # Define class names
        MetadataCatalog.get("icra_realtime").thing_classes = [
            "1-Cheez-it", "2-Starkist_Tuna", "3-Scissors", "4-Frenchs_Mustard", "5-Tomato_Soup",
            "6-Foam_Brick", "7-Clamp", "8-Plastic_Banana", "9-Mug", "10-meat_can",
            "31-Plastic_White_Cup", "32-Wine_Glass", "33-Key", "34-Nail", "35-Laugh_Out_Loud_Joke_Book",
            "36-Adjustable_Wrench", "37-T-shirt", "38-Rolodex_Jumbo_Pencil_Cup", "39-Glove", "40-Pringles"
        ]
        self.metadata = MetadataCatalog.get("icra_realtime")
        self.predictor = DefaultPredictor(cfg)

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("/your_camera_topic", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.result_pub = rospy.Publisher("/detection_result", DetectionResult, queue_size=10)
        rospy.loginfo("Instance segmentation node ready and waiting for images.")

    def image_callback(self, msg):
        rospy.loginfo("Image received. Running inference...")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            outputs = self.predictor(cv_image)
            instances = outputs["instances"].to("cpu")

            result_msg = DetectionResult()
            result_msg.header = Header()
            result_msg.header.stamp = rospy.Time.now()
            result_msg.height = cv_image.shape[0]
            result_msg.width = cv_image.shape[1]

            for i in range(len(instances)):
                class_id = int(instances.pred_classes[i])
                score = float(instances.scores[i])
                label = self.metadata.thing_classes[class_id]
                mask = instances.pred_masks[i].numpy().astype(np.uint8) * 255

                instance_msg = DetectionInstance()
                instance_msg.label = label
                instance_msg.score = score
                instance_msg.mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")

                result_msg.instances.append(instance_msg)

            self.result_pub.publish(result_msg)
            rospy.loginfo("Detection results published.")

        except Exception as e:
            rospy.logerr(f"Error during image callback: {e}")

if __name__ == '__main__':
    try:
        node = InstanceSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
