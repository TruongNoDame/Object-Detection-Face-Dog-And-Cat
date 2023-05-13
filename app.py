import streamlit as st
import cv2
import torch
from utils_yolov7.hubconf import custom
from utils_yolov7.plots import plot_one_box
import numpy as np
import tempfile
from PIL import ImageColor, Image
import time
from collections import Counter
import json
import psutil
import subprocess
import pandas as pd
import socket                   
import sys        
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
# from PIL import Image
import glob
# import torch
import os
# import sys
import argparse
from pathlib import Path
# import detect
# from google.colab.patches import cv2_imshow
# from models.experimental import attempt_load
# import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils                  

from collections import OrderedDict
from subprocess import check_output
from threading import Thread    

IMAGE_SIZE = (16, 12) # Output display size as you want
import matplotlib.pyplot as plt
PATH_TO_SAVED_MODEL="saved_model"
# print('Loading model...', end='')
# Load saved model and build the detection function
# detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
# print('Done!')
#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("pets_label_map.pbtxt",use_display_name=True)
#category_index=label_map_util.create_category_index_from_labelmap([path_to_label_map],use_display_name=True)
# def load_image_into_numpy_array(path):

#     return np.array(Image.open(path))

# ==================== Load weights of YoloV5 ============================= #
# yolo_v5 = attempt_load("ultralytics/yolov5","custom",'best.pt')
# yolo_v5 = torch.hub.load("ultralytics/yolov5","custom",'best.pt', force_reload=True)
# yolo_v5.conf = 0.1  # NMS confidence threshold
# yolo_v5.iou = 0.7  # NMS IoU threshold

# ==================== Load weights of YoloV7 ============================= #
# yolo_v7 = attempt_load('best_yolo_v7.pt')
# yolo_v7.conf = 0.1  # NMS confidence threshold
# yolo_v7.iou = 0.7  # NMS IoU threshold


# st.title("Object Detection Streamlit App")
detect = st.container()
faster_rcnn, yolov5, yolov7 = st.columns(3)

def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]

def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[2], color_rgb_list[1], color_rgb_list[0]]
    return color

p_time = 0

# st.title('YOLOv7 Predictions')
# sample_img = cv2.imread('logo.jpg')
# FRAME_WINDOW = st.image(sample_img, channels='BGR')
st.sidebar.title('Settings')

# path to model
# path_model_file = st.sidebar.text_input(
#     'path to YOLOv7 Model:',
#     'eg: dir/yolov7.pt'
# )

path_model_file_yolov7 = "best_yolo_v7.pt"
path_model_file_faster = "saved_model.pb"


# Class txt
path_to_class_txt = st.file_uploader(
    'Class.txt:',
     type=['txt']
)

# ok = True
if path_to_class_txt is not None:
# if ok == True:

    # options = st.sidebar.radio(
    #     'Options:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)
    options = st.sidebar.radio(
        'Options:', ('Image', 'Video'), index=0)

    # gpu_option = st.sidebar.radio(
    #     'PU Options:', ('CPU', 'GPU'))

    # if not torch.cuda.is_available():
    #     st.sidebar.warning('CUDA Not Available, So choose CPU')
    # else:
    #     st.sidebar.success(
    #         'GPU is Available on this Device, Choose GPU for the best performance',
    #         icon="✅"
    #     )

    # Confidence
    # confidence = st.sidebar.slider(
    #     'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)
    confidence = 0.15
    # Draw thickness
    # draw_thick = st.sidebar.slider(
    #     'Draw Thickness:', min_value=1,
    #     max_value=20, value=3
    # )
    draw_thick = 2
    
    # read class.txt
    bytes_data = path_to_class_txt.getvalue()
    class_labels = bytes_data.decode('utf-8').split("\n")
    color_pick_list = []

    for i in range(len(class_labels)):
        classname = class_labels[i]
        color = color_picker_fn(classname, i)
        color_pick_list.append(color)

    # Image
    if options == 'Image':
        upload_img_file = st.sidebar.file_uploader(
            'Upload Image', type=['jpg', 'jpeg', 'png'])
        if upload_img_file is not None:
            pred_yolov7 = st.checkbox('Dog And Cat Face Predict')
            file_bytes = np.asarray(
                bytearray(upload_img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            FRAME_WINDOW = st.image(img, channels='BGR')
            # FRAME_WINDOW.image(img, channels='BGR')

            if pred_yolov7:
                # if gpu_option == 'CPU':
                model_yolov7 = custom(path_or_model=path_model_file_yolov7)
                    # model_faster = custom(path_or_model=path_model_file_faster)
                # if gpu_option == 'GPU':
                #     model_yolov7 = custom(path_or_model=path_model_file_yolov7, gpu=True)
                    # model_faster = custom(path_or_model=path_model_file_faster, gpu=True)

                bbox_list_yolov7 = []
                current_no_class_yolov7 = []
                results_yolov7 = model_yolov7(img)
                # results_faster = model_faster(img)
                
                # Bounding Box
                box_yolov7 = results_yolov7.pandas().xyxy[0]
                class_list_yolov7 = box_yolov7['class'].to_list()

                for i in box_yolov7.index:
                    xmin, ymin, xmax, ymax, conf = int(box_yolov7['xmin'][i]), int(box_yolov7['ymin'][i]), int(box_yolov7['xmax'][i]), \
                        int(box_yolov7['ymax'][i]), box_yolov7['confidence'][i]
                    if conf > confidence:
                        bbox_list_yolov7.append([xmin, ymin, xmax, ymax])
                if len(bbox_list_yolov7) != 0:
                    for bbox_yolov7, id in zip(bbox_list_yolov7, class_list_yolov7):
                        plot_one_box(bbox_yolov7, img, label=class_labels[id],
                                     color=color_pick_list[id], line_thickness=draw_thick)
                        current_no_class_yolov7.append([class_labels[id]])
                FRAME_WINDOW.image(img, channels='BGR')


                # Current number of classes
                class_fq_yolov7 = dict(Counter(i for sub in current_no_class_yolov7 for i in set(sub)))
                class_fq_yolov7 = json.dumps(class_fq_yolov7, indent = 4)
                class_fq_yolov7 = json.loads(class_fq_yolov7)
                df_fq_yolov7 = pd.DataFrame(class_fq_yolov7.items(), columns=['Class', 'Number'])
                st.write("Yolov7")

                
                # Updating Inference results
                # with st.container():
                #     st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                #     st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                #     st.dataframe(df_fq_yolov7, use_container_width=True)
                detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

                def load_image_into_numpy_array(path):

                    return np.array(Image.open(path))

                #faster rcnn vs yolov5
                yolo_v5 = torch.hub.load("ultralytics/yolov5","custom",'best.pt', force_reload=True)
                yolo_v5.conf = 0.1  # NMS confidence threshold
                yolo_v5.iou = 0.7  # NMS IoU threshold

                detect = st.container()
                faster_rcnn, yolov5, yolov7 = st.columns(3)

                image_np = load_image_into_numpy_array(upload_img_file)

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

                    img = Image.open(upload_img_file)
                    results = yolo_v5(img)
                    results.save(f'data/{upload_img_file.name}')

                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.write("Yolo v5")

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
                    st.image(img1, channels='BGR')
                    st.write("Faster R-CNN")


    # Video
    if options == 'Video':
        upload_video_file = st.sidebar.file_uploader(
            'Upload Video', type=['mp4', 'avi', 'mkv'])
        if upload_video_file is not None:
            pred = st.checkbox('Predict Using YOLOv7')
            # Model
            # if gpu_option == 'CPU':
            model = custom(path_or_model=path_model_file_yolov7)
            # if gpu_option == 'GPU':
            #     model = custom(path_or_model=path_model_file, gpu=True)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(upload_video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            # FRAME_WINDOW = st.image(cap, channels='BGR')
            if pred:
                FRAME_WINDOW.image([])
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            'Video file NOT working\n \
                            Check Video path or file properly!!',
                            icon="🚨"
                        )
                        break
                    current_no_class = []
                    bbox_list = []
                    results = model(img)
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id],
                                         color=color_pick_list[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    FRAME_WINDOW.image(img, channels='BGR')
                    
                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    
                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with stframe1.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        if round(fps, 4)>1:
                            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                    
                    with stframe2.container():
                        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq)
                        # , use_container_width=True

                    with stframe3.container():
                        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                        js1, js2, js3 = st.columns(3)                       

                        # Updating System stats
                        with js1:
                            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                            mem_use = psutil.virtual_memory()[2]
                            if mem_use > 50:
                                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

                        with js2:
                            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
                            cpu_use = psutil.cpu_percent()
                            if mem_use > 50:
                                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

                        with js3:
                            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
                            try:
                                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
                            except:
                                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)


    # Web-cam
    if options == 'Webcam':
        cam_options = st.sidebar.selectbox('Webcam Channel',
                                           ('Select Channel', '0', '1', '2', '3'))
        # Model
        # if gpu_option == 'CPU':
        model = custom(path_or_model=path_model_file)
        # if gpu_option == 'GPU':
        #     model = custom(path_or_model=path_model_file, gpu=True)

        if len(cam_options) != 0:
            if not cam_options == 'Select Channel':
                cap = cv2.VideoCapture(int(cam_options))
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f'Webcam channel {cam_options} NOT working\n \
                            Change channel or Connect webcam properly!!',
                            icon="🚨"
                        )
                        break

                    bbox_list = []
                    current_no_class = []
                    results = model(img)
                    
                    # Bounding Box
                    box = results.pandas().xyxy[0]
                    class_list = box['class'].to_list()

                    for i in box.index:
                        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                            int(box['ymax'][i]), box['confidence'][i]
                        if conf > confidence:
                            bbox_list.append([xmin, ymin, xmax, ymax])
                    if len(bbox_list) != 0:
                        for bbox, id in zip(bbox_list, class_list):
                            plot_one_box(bbox, img, label=class_labels[id],
                                         color=color_pick_list[id], line_thickness=draw_thick)
                            current_no_class.append([class_labels[id]])
                    FRAME_WINDOW.image(img, channels='BGR')

                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                    
                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    with stframe1.container():
                        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                        if round(fps, 4)>1:
                            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                    
                    with stframe2.container():
                        st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                        st.dataframe(df_fq, use_container_width=True)

                    with stframe3.container():
                        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                        js1, js2, js3 = st.columns(3)                       

                        # Updating System stats
                        with js1:
                            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                            mem_use = psutil.virtual_memory()[2]
                            if mem_use > 50:
                                js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

                        with js2:
                            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
                            cpu_use = psutil.cpu_percent()
                            if mem_use > 50:
                                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
                            else:
                                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

                        with js3:
                            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
                            try:
                                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
                            except:
                                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)


    # RTSP
    if options == 'RTSP':
        rtsp_url = st.sidebar.text_input(
            'RTSP URL:',
            'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
        )
        # st.sidebar.markdown('Press Enter after pasting RTSP URL')
        url = rtsp_url[:-11]
        rtsp_options = st.sidebar.selectbox(
            'RTSP Channel',
            ('Select Channel', '0', '1', '2', '3',
                '4', '5', '6', '7', '8', '9', '10')
        )

        # Model
        # if gpu_option == 'CPU':
        model = custom(path_or_model=path_model_file)
        # if gpu_option == 'GPU':
        #     model = custom(path_or_model=path_model_file, gpu=True)

        if not rtsp_options == 'Select Channel':
            cap = cv2.VideoCapture(f'{url}{rtsp_options}&subtype=0')
            stframe1 = st.empty()
            stframe2 = st.empty()
            stframe3 = st.empty()
            while True:
                success, img = cap.read()
                if not success:
                    st.error(
                        f'RSTP channel {rtsp_options} NOT working\nChange channel or Connect properly!!',
                        icon="🚨"
                    )
                    break

                bbox_list = []
                current_no_class = []
                results = model(img)
                
                # Bounding Box
                box = results.pandas().xyxy[0]
                class_list = box['class'].to_list()

                for i in box.index:
                    xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
                        int(box['ymax'][i]), box['confidence'][i]
                    if conf > confidence:
                        bbox_list.append([xmin, ymin, xmax, ymax])
                if len(bbox_list) != 0:
                    for bbox, id in zip(bbox_list, class_list):
                        plot_one_box(bbox, img, label=class_labels[id],
                                     color=color_pick_list[id], line_thickness=draw_thick)
                        current_no_class.append([class_labels[id]])
                FRAME_WINDOW.image(img, channels='BGR')

                # FPS
                c_time = time.time()
                fps = 1 / (c_time - p_time)
                p_time = c_time
                
                # Current number of classes
                class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                class_fq = json.dumps(class_fq, indent = 4)
                class_fq = json.loads(class_fq)
                df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                
                # Updating Inference results
                with stframe1.container():
                    st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                    if round(fps, 4)>1:
                        st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h4 style='color:red;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
                
                with stframe2.container():
                    st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                    st.dataframe(df_fq, use_container_width=True)

                with stframe3.container():
                    st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
                    js1, js2, js3 = st.columns(3)                       

                    # Updating System stats
                    with js1:
                        st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
                        mem_use = psutil.virtual_memory()[2]
                        if mem_use > 50:
                            js1_text = st.markdown(f"<h5 style='color:red;'>{mem_use}%</h5>", unsafe_allow_html=True)
                        else:
                            js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

                    with js2:
                        st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
                        cpu_use = psutil.cpu_percent()
                        if mem_use > 50:
                            js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
                        else:
                            js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

                    with js3:
                        st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
                        try:
                            js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
                        except:
                            js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)
