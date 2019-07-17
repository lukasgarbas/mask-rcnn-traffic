from mrcnn import utils
from mrcnn import model as modellib
from coco import coco
import cv2
import os
import argparse
import datetime
import csv
import time
import numpy as np
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(
    description='Mask R-CNN for traffic detection')
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
ap.add_argument("-w", "--weights", required=True,
    help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
    help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Loading The Model
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(args['weights'], by_name=True)

# Classes that will be displayed
relevant_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck'
]

# All classes in COCO
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Function to display detected instances and save detections to csv file
def display_instances(frame_num, image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        # Only visualize relevant objects
        if label in relevant_class_names:
            image = apply_mask(image, mask, color)
                      
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

            image = cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Find the centroid of the mask
            mask2 = mask.astype('uint8') * 255
            cnts = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
                M = cv2.moments(c)
                if (M['m00'] == 0.0):
                    cX = cY = 0
                else:
                    # Centroid coordinates
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
            
            centroid = (cX,cY)

            # draw the center of the mask
            image = cv2.circle(image, centroid, 5, (0,0,255), -1)
            
            # saving results (box coordinates and a center point of binary mask) to .csv file
            with open(output_boxes, mode='a') as csv_file:
                w = csv.DictWriter(csv_file, fieldnames=csv_fields)
                entry = {
					'frame' : frame_num,
					'class' : label,
					'left' : x1,
					'top' : y1,
					'right' : x2,
					'bottom' : y2,
					'center_left' : centroid[0],
					'center_top' : centroid[1],
					'confidence' : '{0:.2f}'.format(score)
				}
                w.writerow(entry)
    return image

def random_colors(N, bright=True):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    #set colors for car and person
    colors[3] = (60,179,113)
    colors[1] = (255, 166, 76)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

# Capture the input video
capture = cv2.VideoCapture(args['video'])
total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output_file_name = "traffic_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
output = cv2.VideoWriter(output_file_name, codec, 15.0, size)

first_frame = True
frame_num = 1

# save detected boxes to csv file
output_boxes = "detections_{:%Y%m%dT%H%M%S}.csv".format(datetime.datetime.now())
csv_fields = ['frame', 'class', 'left', 'top', 'right', 
                'bottom', 'center_left', 'center_top', 'confidence']

# Prepare colors for each class label
colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}

with open(output_boxes, mode='w') as csv_file:
    w = csv.DictWriter(csv_file, fieldnames=csv_fields)
    w.writeheader()

# Open the video and run the model on each frame
while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        # add mask to frame
        start = time.time()
        results = model.detect([frame], verbose=0)
        
        r = results[0]

        frame = display_instances(
            frame_num, frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        end = time.time()

        # some information on processing a single frame
        if first_frame:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print(total)
            print("[INFO] estimated total time to finish: {:.4f}".format((elap * total)))
            
        frame_num += 1
        output.write(frame)
    else:
        print('ERROR CAPTURING THE FRAME')
        break

output.release()
capture.release()
