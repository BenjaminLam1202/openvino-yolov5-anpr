#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp 
from math import sqrt
from time import time
import numpy as np
import random
import string
import datetime
import uuid
import cmath as math
import multiprocessing as mp

import cv2
from openvino.inference_engine import IENetwork, IECore

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()
prev_x, prev_y = None, None


class CarTracker:
    def __init__(self, bbox):
        self.id = uuid.uuid4().hex[:6].upper()  # Generate a unique ID for the tracker
        self.kf = cv2.KalmanFilter(4, 2)  # Initialize a Kalman filter with 4 state variables and 2 measurement variables
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)  # Set the measurement matrix to extract position information
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # Set the transition matrix to update position and velocity
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]], np.float32)  # Set the process noise covariance matrix to account for measurement noise and acceleration
        self.kf.statePost = np.array([[bbox['xmin']], [bbox['ymin']], [0], [0]], np.float32)  # Initialize the state vector to the position of the current bbox
        self.last_detection = time()  # Record the time of the last detection

    def update(self, bbox):
        measurement = np.array([[bbox['xmin']], [bbox['ymin']]], np.float32)  # Extract the position information from the current bbox
        self.kf.correct(measurement)  # Update the Kalman filter with the measurement
        self.last_detection = time()  # Record the time of the last detection

    def predict(self):
        self.kf.predict()
        predicted_bbox = {'xmin': int(self.kf.statePost[0, 0]), 'ymin': int(self.kf.statePost[1, 0]), 'xmax': int(self.kf.statePost[0, 0] + (self.kf.statePost[2, 0],  10)), 'ymax': int(self.kf.statePost[1, 0] + (self.kf.statePost[3, 0],  10))}
        return predicted_bbox




def is_same_object(curr, pre_list, threshold=100):
    # Calculate the center point of curr
    curr_center_x = (curr['xmin'] + curr['xmax']) / 2
    curr_center_y = (curr['ymin'] + curr['ymax']) / 2
    if pre_list == []:
        return False
    # Check if there are any objects in pre_list whose center point is close to curr
    for pre in pre_list:
        pre_center_x = (pre['xmin'] + pre['xmax']) / 2
        pre_center_y = (pre['ymin'] + pre['ymax']) / 2
        A = (curr_center_x - pre_center_x)**2
        B = (curr_center_y - pre_center_y)**2
        cur = A + B
        distance = sqrt(cur)
        # print(distance)
        if distance < threshold:
            return True
    return False

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=list)
    args.add_argument("-oj","--objects", help="Optional. object", default=[2], type=list)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument("--show", help="Optional. Don't show output",action='store_true')
    return parser


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self,  side):
        self.num = 3 #if 'num' not in param else int(param['num'])
        self.coords = 4 #if 'coords' not in param else int(param['coords'])
        self.classes = 80 #if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] #if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        #self.isYoloV3 = False

        #if param.get('mask'):
        #    mask = [int(idx) for idx in param['mask'].split(',')]
        #    self.num = len(mask)

        #    maskedAnchors = []
        #    for idx in mask:
        #        maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        #    self.anchors = maskedAnchors

        #    self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        # [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

def resize_(image, scale):
    """
    Compress cv2 resize function
    """
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dsize = (width, height)
    return cv2.resize(image, dsize,interpolation=cv2.INTER_AREA)

def detect_and_save_car(frame, detection_result, save_dir='car_images/', threshold_distance=100, confidence_threshold=0.5):
    global prev_x, prev_y

    # Extract the bounding box coordinates, confidence level, and class ID from the detection result
    xmin, xmax, ymin, ymax = detection_result['xmin'], detection_result['xmax'], detection_result['ymin'], detection_result['ymax']
    confidence = detection_result['confidence']
    class_id = detection_result['class_id']

    # Calculate the x and y coordinates of the center of the bounding box
    x, y = (xmin + xmax) // 2, (ymin + ymax) // 2

    # Check if car is detected with confidence above threshold
    if confidence > confidence_threshold:

        # Check if previous position of car is not set or the car has moved a distance above threshold_distance
        if prev_x is None or prev_y is None or abs(x - prev_x) > threshold_distance or abs(y - prev_y) > threshold_distance:

            # Save image with current timestamp
            filename = f"{format_date(detection_result)}" 
            filepath = os.path.join(save_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            saved_yet = cv2.imwrite(filepath, frame)
            print(f"save {filepath} {saved_yet}")

            # Update previous position of car
            prev_x, prev_y = x, y


def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str

def format_date(detection_result):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y/%m/%d/%H")
    file_str = now.strftime(f"%y_%m_%d_%H_%M_%S_{detection_result['class_id']}_{int(float(detection_result['confidence'])*100)}.jpg")
    return f"{date_str}/{file_str}"

def save_image_with_date_format(image, directory):
    filename = format_date()
    filepath = os.path.join(directory, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    cv2.imwrite(filepath, image)
    return filepath

def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h=640, resized_im_w=640):
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2  # wh padding
    x = int((x - pad[0])/gain)
    y = int((y - pad[1])/gain)

    w = int(width/gain)
    h = int(height/gain)
 
    xmin = max(0, int(x - w / 2))
    ymin = max(0, int(y - h / 2))
    xmax = min(im_w, int(xmin + w))
    ymax = min(im_h, int(ymin + h))
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------    
    out_blob_n, out_blob_c, out_blob_h, out_blob_w = blob.shape
    predictions = 1.0/(1.0+np.exp(-blob)) 
                   
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
 
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = int(out_blob_c/params.num) #4+1+num_classes

    for row, col, n in np.ndindex(params.side, params.side, params.num):
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        x = (2*x - 0.5 + col)*(resized_image_w/out_blob_w)
        y = (2*y - 0.5 + row)*(resized_image_h/out_blob_h)
        if int(resized_image_w/out_blob_w) == 8 & int(resized_image_h/out_blob_h) == 8: #80x80, 
            idx = 0
        elif int(resized_image_w/out_blob_w) == 16 & int(resized_image_h/out_blob_h) == 16: #40x40
            idx = 1
        elif int(resized_image_w/out_blob_w) == 32 & int(resized_image_h/out_blob_h) == 32: # 20x20
            idx = 2

        width = (2*width)**2* params.anchors[idx * 6 + 2 * n]
        height = (2*height)**2 * params.anchors[idx * 6 + 2 * n + 1]
        class_id = np.argmax(class_probabilities)
        confidence = object_probability
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, resized_im_h=resized_image_h, resized_im_w=resized_image_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

# Define a function to generate a unique ID for each car
def generate_car_id(bbox):
    xmin, ymin, xmax, ymax = bbox
    return str(xmin) + "_" + str(ymin) + "_" + str(xmax) + "_" + str(ymax)

def is_near_center(obj, screen_width, screen_height, threshold):
    center_x = screen_width / 2
    center_y = screen_height / 2
    box_center_x = (obj['xmin'] + obj['xmax']) / 2
    box_center_y = (obj['ymin'] + obj['ymax']) / 2
    distance_x = abs(box_center_x - center_x)
    distance_y = abs(box_center_y - center_y)
    return distance_x < threshold  * screen_width and distance_y < threshold * screen_height

def process_object(arg):
    in_frame, obj, pre_list, origin_im_size, labels_map, args, detection_object_list, frame = arg

    curr = obj
    in_frame_shape = in_frame.shape[2:]
    check = is_same_object(curr, pre_list, threshold=100)
    check_center = is_near_center(curr, in_frame_shape[0], in_frame_shape[1], 100)

    # Validation bbox of detected object
    valid_box = np.logical_and.reduce((
        np.greater_equal(obj['xmin'], 0),
        np.less(obj['xmin'], obj['xmax']),
        np.less(obj['xmax'], origin_im_size[1]),
        np.greater_equal(obj['ymin'], 0),
        np.less(obj['ymin'], obj['ymax']),
        np.less(obj['ymax'], origin_im_size[0])
    ))
    if not valid_box:
        return

    color = (min(int(obj['class_id'] * 12.5), 255), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
    det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(obj['class_id'])

    log_message = "{:^9} | {:^10f} | {:^4} | {:^4} | {:^4} | {:^4} | {}".format(
        det_label, obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], color)
    if args.raw_output_message:
        log.info(log_message)
    
    if obj['class_id'] in detection_object_list:
        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        cv2.putText(frame, "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        if check == False and obj['confidence'] >= 0.70 and check_center == True:
            detect_and_save_car(frame, detection_result=obj)
            start_time = time()


def main():
    args = build_argparser().parse_args()


    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    model = args.model
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model)

    assert len(net.input_info.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
    input_blob = next(iter(net.input_info))

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    input_stream = 0 if args.input == "cam" else args.input

    is_async_mode = True
    cap = cv2.VideoCapture(input_stream)
    number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    wait_key_code = 1

    # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
    if number_input_frames != 1:
        ret, frame = cap.read()
    else:
        is_async_mode = False
        wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # Define a maximum distance threshold
    max_distance = 10  # adjust this value based on your use case
    pre_list = []
    # Initialize the list of previous object IDs and their corresponding center points or bounding boxes
    prev_objects = []
    detection_object_list = args.objects
    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    while True:
        # Capture frame and check for errors
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Preprocess frame
        in_frame = letterbox(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        
        # Start inference
        start_time = time()
        exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
        det_time = time() - start_time
        
        # Collect object detection results
        objects = list()
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            output = exec_net.requests[cur_request_id].output_blobs
            for layer_name, out_blob in output.items():
                layer_params = YoloParams(side=out_blob.buffer.shape[2])
                objects += parse_yolo_region(out_blob.buffer, in_frame.shape[2:], frame.shape[:-1], layer_params, args.prob_threshold)
        
        # Filter overlapping boxes
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                    objects[j]['confidence'] = 0
        objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

        # Initialize trackers and tracker IDs
        origin_im_size = frame.shape[:-1]

        # Process objects in parallel
        arg_list = [(in_frame, obj, pre_list, origin_im_size, labels_map, args, detection_object_list, frame) for obj in objects]
        pool = mp.Pool(processes=10)
        result = pool.map(process_object,arg_list)
        pre_list = objects

        # Show results if desired

        cv2.imshow("TonVision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()




if __name__ == '__main__':
    sys.exit(main() or 0)
