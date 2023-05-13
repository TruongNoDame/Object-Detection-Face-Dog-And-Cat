#Loading the saved_model
import tensorflow as tf
import time
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import glob
import torch
import os
import sys
import argparse
from pathlib import Path
# import detect
# from google.colab.patches import cv2_imshow
# from models.experimental import attempt_load
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# ==================== Load weights of Faster-RCNN ============================= #
IMAGE_SIZE = (12, 8) # Output display size as you want
import matplotlib.pyplot as plt
PATH_TO_SAVED_MODEL="saved_model"
print('Loading model...', end='')
# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')
#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("pets_label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)
def load_image_into_numpy_array(path):

    return np.array(Image.open(path))

# ==================== Load weights of YoloV5 ============================= #
# yolo_v5 = attempt_load("ultralytics/yolov5","custom",'best.pt')
yolo_v5 = torch.hub.load("ultralytics/yolov5","custom",'best.pt')
yolo_v5.conf = 0.1  # NMS confidence threshold
yolo_v5.iou = 0.7  # NMS IoU threshold

# ==================== Load weights of YoloV7 ============================= #
# yolo_v7 = attempt_load('best_yolo_v7.pt')
# yolo_v7.conf = 0.1  # NMS confidence threshold
# yolo_v7.iou = 0.7  # NMS IoU threshold


st.title("Object Detection Streamlit App")
detect = st.container()
faster_rcnn, yolov5, yolov7 = st.columns(3)

with detect:
    uploaded_file = st.file_uploader("Chọn ảnh đi bạn ei !!!")
    if uploaded_file is not None:
        # In ra 5 điểm dữ liệu đầu tiên
        image_np = load_image_into_numpy_array(uploaded_file)

        with faster_rcnn:
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.4, # Adjust this value to set the minimum probability boxes to be classified as True
                agnostic_mode=False)

            img1 = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
            st.image(img1)
            st.write("Faster R-CNN")

        with yolov5:
            def get_subdirs(b='.'):
                result = []
                for d in os.listdir(b):
                    bd = os.path.join(b, d)
                    if os.path.isdir(bd):
                        result.append(bd)
                return result
            def get_detection_folder():
                return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

            img = Image.open(uploaded_file)
            results = yolo_v5(img)
            results.save(f'data/{uploaded_file.name}')

            for img in os.listdir(get_detection_folder()):
                st.image(str(Path(f'{get_detection_folder()}') / img))

            st.write("Yolo v5")
        
        # with yolov7:
        #     def get_subdirs(b='.'):
        #         result = []
        #         for d in os.listdir(b):
        #             bd = os.path.join(b, d)
        #             if os.path.isdir(bd):
        #                 result.append(bd)
        #         return result
        #     def get_detection_folder():
        #         return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

        #     img = Image.open(uploaded_file)
        #     results = detect(weights='best_yolo_v7.pt',conf=0.1,source=uploaded_file)
        #     results.save(f'data/{uploaded_file.name}')

        #     for img in os.listdir(get_detection_folder()):
        #         st.image(str(Path(f'{get_detection_folder()}') / img))

        #     st.write("Yolo v7")
