import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import keras
import keras.models as KM
import keras.layers as KL
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K



from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os
import json

import numpy as np
from tqdm import tqdm

import gc
import imblearn
from xml.etree import ElementTree
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import random
import math

import numpy as np
from numpy import array
from os import listdir
import os

import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model

# %matplotlib inline

from os import listdir
from xml.etree import ElementTree
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import sys

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import warnings
warnings.filterwarnings('ignore')

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from PIL import Image
import PIL
import glob
import numpy as np
from sklearn import metrics

print("################################Imports Done################################")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for training')
parser.add_argument('--weight_name', dest='weight_name', type=str, help='name of weights')
args = parser.parse_args()

weight_name = args.weight_name
epochs = args.epochs
base_path = '/content/drive/MyDrive/Workspace'
results_path = f"{base_path}/results/{weight_name}/after"

if not os.path.exists(f"{base_path}/results"):
   os.mkdir(f"{base_path}/results")

if not os.path.exists(f"{base_path}/results/{weight_name}"):
   os.mkdir(f"{base_path}/results/{weight_name}")

if not os.path.exists(f"{base_path}/results/{weight_name}/after"):
   os.mkdir(f"{base_path}/results/{weight_name}/after")

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    

def TopK(x, k):
    a = dict([(i, j) for i, j in enumerate(x)])
    sorted_a = dict(sorted(a.items(), key = lambda kv:kv[1], reverse=True))
    indices = list(sorted_a.keys())[:k]
    values = list(sorted_a.values())[:k]
    return (indices, values)

def class_val_func2():
  imgpath = base_path + '/Objects-Dataset/images/'
  annpath = base_path + '/Objects-Dataset/annots/'

  n = 0
  mae = 0 
  mse = 0
  mses = []

  for filename in listdir(imgpath):
    image_id = filename.split('.')[0]
    
    n += 1

    img= plt.imread(imgpath + str(image_id) + ".jpg")
    result = mr_model_inf.detect([img])
    r = result[0]

    tree = ET.parse(annpath + str(image_id) +'.xml')
    root = tree.getroot()

    distance = float(root.find('./distance').text)

    y = r['distance'][0]
    mae += abs(y-distance)
    mse += pow(y-distance,2)
    mses.append(pow(y-distance,2))
    
  
  return {
      "total": n,
      "mae": mae / n,
      "mse": mse / n,
      "mses": mses
  }
    
    
    
def custom_acc():
  imgpath = base_path + '/Objects-Dataset/images/'
  annpath = base_path + '/Objects-Dataset/annots/'

  n = 0
  s1 = 0 
  s2 = 0

  for filename in listdir(imgpath):
    image_id = filename.split('.')[0]
    
    n += 1

    img= plt.imread(imgpath + str(image_id) + ".jpg")
    result = mr_model_inf.detect([img])
    r = result[0]

    tree = ET.parse(annpath + str(image_id) +'.xml')
    root = tree.getroot()

    distance = float(root.find('./distance').text)

    y = r['distance'][0]
    if abs(y-distance) < 30:
      s1 += 1
    if abs(y-distance) < 15:
      s2 += 1
    
  
  return {
      "total": n,
      "s1": s1 / n,
      "s2": s2 / n,
  }


def class_val_func():
  imgpath = base_path + '/Objects-Dataset/images/'
  annpath = base_path + '/Objects-Dataset/annots/'

  n = 0
  mae = 0 
  mse = 0

  for filename in listdir(imgpath):
    image_id = filename.split('.')[0]
    
    n += 1

    img= plt.imread(imgpath + str(image_id) + ".jpg")
    result = mr_model_inf.detect([img])
    r = result[0]

    tree = ET.parse(annpath + str(image_id) +'.xml')
    root = tree.getroot()

    distance = float(root.find('./distance').text)

    y = r['distance'][0]
    mae += abs(y-distance)
    mse += pow(y-distance,2)
    
  
  return {
      "total": n,
      "mae": mae / n,
      "mse": mse / n,
  }
    
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou   


plt.rcParams["font.serif"] = "Times New Roman"
csfont = {'fontname':'DejaVu Serif'}
def visualize2(imageID):
#   imgpath = base_path + '/Test-Images/'
#   annpath = base_path + '/Test-Images/'
  imgpath = base_path + '/Objects-Dataset/images/'
  annpath = base_path + '/Objects-Dataset/annots/'

  img= plt.imread(imgpath + str(imageID) + ".jpg")
  plt.figure(figsize=(5, 5),dpi=600)
  plt.imshow(img)
  result = mr_model_inf.detect([img])
  r = result[0]

  classes = r['class_ids']-1

  print(classes)

  pn1 = 2-len(classes)
  for i in range(0,pn1):
    classes=np.vstack([classes,11])
  print(classes)

  classNames = ['','']

  if classes[0] ==0:
      classNames[0] = 'balloon'
  elif classes[0] ==1:
      classNames[0] = 'bike'
  elif classes[0] ==2:
      classNames[0] = 'camel'
  elif classes[0] ==3:
      classNames[0] = 'car'
  elif classes[0] ==4:
      classNames[0] = 'helicopter'
  elif classes[0] ==5:
      classNames[0] = 'ladder'
  elif classes[0] ==6:
      classNames[0] = 'sofa'
  else:
      classNames[0] = 'Unknown'

  if classes[1] ==0:
      classNames[1] = 'balloon'
  elif classes[1] ==1:
      classNames[1] = 'bike'
  elif classes[1] ==2:
      classNames[1] = 'camel'
  elif classes[1] ==3:
      classNames[1] = 'car'
  elif classes[1] ==4:
      classNames[1] = 'helicopter'
  elif classes[1] ==5:
      classNames[1] = 'ladder'
  elif classes[1] ==6:
      classNames[1] = 'sofa'
  else:
      classNames[1] = 'Unknown'


  tree = ET.parse(annpath + str(imageID) +'.xml')
  root = tree.getroot()

  disLabels = ['Very Close','Close','Far','Too Far']

  distance = float(root.find('./distance').text)
  distanceLabel = int(root.find('./distanceLabel').text)
  # plt.title(f" Actual Label:{disLabels[distanceLabel]},            Predicted Label:{disLabels[r['distance_label']]} \n ActualDistance:{distance:.2f}, PredictedDistance:{r['distance']:.2f}",loc='left', fontsize=9)

  aboxes = []
  for box in root.findall('.//bndbox'):
    xmin = int(box.find('xmin').text)
    ymin = int(box.find('ymin').text)
    xmax = int(box.find('xmax').text)
    ymax = int(box.find('ymax').text)
    coors = [xmin, ymin, xmax, ymax]
    aboxes.append(coors)

  anames = []
  for name in root.findall('.//name'):
    anames.append(name.text)


  ax = plt.gca()
  rects=[]
  for box in aboxes:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height, fill=False, color='green')
    rects.append(rect)
    ax.add_patch(rect)
  for i, box in enumerate(r['rois']):
    y1, x1, y2, x2 = box
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height, fill=False, color='red', linestyle='--')
    rects.append(rect)
    if i in range (0,2):
      ax.annotate(classNames[i], (x1+height - 15, y1-6), color='red', fontsize=5, ha='center', va='center',**csfont)
    else:
      ax.annotate('Unknown', (x1+height - 15, y1-6), color='red', fontsize=5, ha='center', va='center',**csfont)

    ax.add_patch(rect)
  # ax.legend([rects[0],rects[2]], ['ActualBox','PredictedBox'])
  # plt.text(5, 440,f" Actual Label:{disLabels[distanceLabel]}, Predicted Label:{disLabels[r['distance_label']]}\n \n ActualDistance:{distance:.2f}, PredictedDistance:{r['distance']:.2f}",\
          #  bbox=dict(fill=False, linewidth=0), fontsize=6,**csfont )
    plt.title(f"Actual: {distance:.2f}[{disLabels[distanceLabel]}] - Predicted: {r['distance']:.2f}[{disLabels[r['distance_label']]}]")
  plt.axis('off')
  plt.savefig(f"{results_path}/{imageID}.jpg")
  plt.close()
              
              

def evaluate_model(dataset, model):
  imgpath = base_path + '/Objects-Dataset/images/'
  annpath = base_path + '/Objects-Dataset/annots/'

  n = 0
  mae = 0
  mse = 0
  accuracy = 0
  class_accuracy = 0
  acc_ious = []
  a_allboxes = []
  p_allboxes = []
  iousum=0

  for image_id in dataset.image_ids:

    img_info = dataset.image_info[image_id]
    img= plt.imread(img_info['path'])
    result = mr_model_inf.detect([img])
    r = result[0]
    # print(r['class_ids'],r['rois'],r['distance_label'],r['distance'])
    pnames = []
    pnames = r['class_ids'] - 1
    scores = r['scores']
    n += 1
    # print("LEN",len(pnames),pnames)
    # if len(pnames)==0:
    #   pname = [9,9]
    # elif len(pnames)==1:
    #   pnames[1]=9
    if len(pnames)<2:
      pnames = np.full(2-len(pnames),9)
    
    # print("LEN",len(pnames),pnames)


    tree = ET.parse(img_info['annotation'])
    root = tree.getroot()
    filename = str(root.find('./filename').text)[0:5]

    names = []
    for name in root.findall('./object/name'):
      names.append(int(name.text))

    if ((int(pnames[0])==names[0]) and int(pnames[1])==names[1]) or ((int(pnames[0])==names[1]) and int(pnames[1])==names[0]) :
      class_accuracy +=2

    aboxes = []
    for box in root.findall('.//bndbox'):
      xmin = int(box.find('xmin').text)
      ymin = int(box.find('ymin').text)
      xmax = int(box.find('xmax').text)
      ymax = int(box.find('ymax').text)
      coors = [xmin, ymin, xmax, ymax]
      a_allboxes.append(coors)
      aboxes.append(coors)

    pboxes = r['rois']
    pn = 2-len(pboxes)
    for i in range(0,pn):
      pboxes=np.vstack([pboxes,np.array([0,0,0,0])])

    pboxes[0][0],pboxes[0][1] = pboxes[0][1],pboxes[0][0]
    pboxes[0][2],pboxes[0][3] = pboxes[0][3],pboxes[0][2]
    pboxes[1][0],pboxes[1][1] = pboxes[1][1],pboxes[1][0]
    pboxes[1][2],pboxes[1][3] = pboxes[1][3],pboxes[1][2]

    if pboxes[0][0] > pboxes[1][0] and pboxes[0][1] > pboxes[1][1]:
      pboxes[[0,1]] = pboxes[[1,0]]
    p_allboxes.append(pboxes[0])
    p_allboxes.append(pboxes[1])


    iou1 = bb_intersection_over_union(pboxes[0],aboxes[0])
    iou2 = bb_intersection_over_union(pboxes[1],aboxes[1])
    acc_ious.append(iou1)
    acc_ious.append(iou2)


    distance = float(root.find('./distance').text)
    distanceLabel = int(root.find('./distanceLabel').text)

    if distanceLabel == int(r['distance_label']):
      accuracy += 1
    mae += abs(distance- float(r['distance']))
    mse += pow(distance- float(r['distance']),2)
  
  result_dict = {
    "number of samples": n,
    "distance label accuracy": accuracy / n,
    "distance mae": mae/n,
    "distance mse": mse/n,
    "class accuracy": class_accuracy / (2*n),
    "bbox IoU": np.mean(acc_ious)
  }

  f = open(f"{base_path}/results/{weight_name}/results.txt", "a")
  f.write(str(result_dict))
  f.write("\n")
  f.close()

  print(result_dict)


print("################################Building Model################################")

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
     # objects + BG
    NUM_CLASSES = 8
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 20
    
    # Learning rate
    LEARNING_RATE=0.001
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

    IMAGE_MAX_DIM = 480 * 2

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1,
        "d_output_loss": 1,
        "d_output_lbl_loss": 1,
    }
  

config= myMaskRCNNConfig()
mr_model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')


print("################################Loading dataset################################")


class Creating_Dataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        
        # Add classes. We have only one class to add.
        self.add_class("dataset", 0, "0")
        self.add_class("dataset", 1, "1")
        self.add_class("dataset", 2, "2")
        self.add_class("dataset", 3, "3")
        self.add_class("dataset", 4, "4")
        self.add_class("dataset", 5, "5")
        self.add_class("dataset", 6, "6")



        self.dataset_dir = dataset_dir
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            image_id = filename.split('.')[0]

            # large
            if is_train and (int(image_id) % 120) > 90:
                continue
            if not is_train and (int(image_id) % 120) < 90:
                continue
            
            # setting image file
            img_path = images_dir + filename
            
            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        classes = list()
        for box in root.findall('.//object'):
            xmin = int(box.find('.//bndbox/xmin').text)
            ymin = int(box.find('.//bndbox/ymin').text)
            xmax = int(box.find('.//bndbox/xmax').text)
            ymax = int(box.find('.//bndbox/ymax').text)
            class_name = box.find('name').text
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            classes.append(class_name)
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        distance = float(root.find('./distance').text)

        return boxes, width, height, classes

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        
        # load XML
        boxes, w, h, classes = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(classes[i]))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    #Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']
    

# prepare train set
train_set = Creating_Dataset()
train_set.load_dataset(f"{base_path}/Objects-Dataset", is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = Creating_Dataset()
test_set.load_dataset(f"{base_path}/Objects-Dataset", is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

print("################################Loading before Weight################################")

model_path = f'{base_path}/Modeling-Autism/AllDistanceLayerWieghts_{weight_name}-BeforeTraining.h5'
mr_model.load_weights(model_path, by_name=True)

print("################################Building inf model################################")
mr_model_inf = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
mr_model_inf.load_weights(model_path, by_name=True)

print("################################Training the model################################")
history = mr_model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=epochs, layers='heads')

print("################################History################################")
print(history)

print("################################Saving weights after training################################")
model_path = f'{base_path}/Modeling-Autism/AllDistanceLayerWieghts_{weight_name}-AfterTraining.h5'
mr_model.keras_model.save_weights(model_path)

print("################################Loading weights after training for inf model################################")
# mr_model.keras_model.save_weights(model_path)
mr_model_inf = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
mr_model_inf.load_weights(model_path, by_name=True)

print("################################Evaluating################################")
evaluate_model(test_set, mr_model_inf)

print("################################Visualizing################################")
images = ['00000','00150','00351','00451']
for i in images:
  visualize2(i)
