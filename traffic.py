"""
Train on the traffic dataset.
------------------------------------------------------------
Usage: import the module or run from the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 traffic.py train --dataset=/path/to/project/dataset --weights=path/to/coco/weights
    # Resume training a model that you had trained earlier
    python3 traffic.py train --dataset=/path/to/project/dataset --weights=path/to/last/weights
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
import cv2
import csv
# from visualize_traffic import model, display_instances, class_names

# Root directory of the project
ROOT_DIR = os.path.abspath("/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save new weights, logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class TrafficConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "traffic"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80 # COCO has 80 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100


############################################################
#  Dataset
############################################################

class TrafficDataset(utils.Dataset):

    def load_traffic(self, dataset_dir, subset):
        """Load train or val subset of the traffic dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add all classes from coco dataset that the pre-trained model had
        with open('object_detection_classes_coco.txt') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        label_id = 1

        for i in content:
            self.add_class("traffic", label_id, i)
            label_id = label_id + 1

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Images for training and validation are stored in dataset_dir
        train_images = glob.glob('{}/images/*.jpg'.format(dataset_dir))

        # Bounding boxes are saved in csv format 
        csv_path = '{}/bounding_boxes.csv'.format(dataset_dir)
        with open(csv_path) as f:
            a = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]  
        bounding_boxes_csv = a

        # Loading images and bounding boxes
        for a in train_images:
            image_id = a.replace('{}/images/'.format(dataset_dir), "")
            image = cv2.imread(train_images[0])
            height, width = image.shape[:2]
            bounding_boxes = []

            for b in bounding_boxes_csv:
                frame_number = image_id.replace(subset, '')
                frame_number = image_id.replace('frame', '')
                frame_number = frame_number.replace('.jpg', '')
                if b['frame'] == frame_number:
                    bounding_boxes.append(b)

            self.add_image(
                "traffic",
                image_id= image_id,  # use file name as a unique image id
                path=a,
                width=width, height=height,
                bounding_boxes = bounding_boxes)
            
            #print("added image: {}, {}, {}, {}".format(image_id, a, width, height))

    def load_mask(self, image_id):
        """Loads masks from PNG files in masks folder.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a traffic dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "traffic":
            return super(self.__class__, self).load_mask(image_id)
        
  
        # Extract all different masks inside an image
        info = self.image_info[image_id]

        masks = np.zeros([info["height"], info["width"], len(info["bounding_boxes"])],
                        dtype=np.uint8)

        class_ids = []
        instance_masks = []

        # Masks are stored in the the subdirectory masks
        mask_path = info["path"].replace('images', 'masks')
        mask_path = mask_path.replace('.jpg', '.png')

        # Load annotated masks in PNG format (convert to grayscale)
        annotated_mask = cv2.imread(mask_path)
        annotated_mask = cv2.cvtColor(annotated_mask, cv2.COLOR_BGR2GRAY)

        # Labels for each class (same ids as in COCO)
        labels = { 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'bus': 6, 'train': 7, 'truck': 8}

        print('preparing: {}'.format(mask_path))

        for i, b in enumerate(info['bounding_boxes']):

            x1 = int(float(b['left']))
            y1 = int(float(b['top']))
            x2 = int(float(b['right']))
            y2 = int(float(b['bottom']))
            class_name = b['class']

            m = np.zeros([info["height"], info["width"]], dtype=np.uint8)
            m[y1:y2, x1:x2] = annotated_mask[y1:y2, x1:x2]

            m = m > 0
            class_ids.append(labels[class_name])
            instance_masks.append(m)

        masks = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return masks.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "traffic":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TrafficDataset()
    dataset_train.load_traffic(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrafficDataset()
    dataset_val.load_traffic(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Training 7 classes in Mask R-CNN COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TrafficConfig()
    else:
        class InferenceConfig(TrafficConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Load weights
    weights_path = args.weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Train the model
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. Use 'train'".format(args.command))