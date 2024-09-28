import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import json
from PIL import Image
import PIL
import glob
import numpy as np


img= plt.imread('/home/el/Documents/My Object Dataset/Dataset/images/00344.jpg')
plt.imshow(img)


tree = ET.parse('/home/el/Documents/My Object Dataset/Dataset/annots/00344.xml')
root = tree.getroot()



distance = float(root.find('./distance').text)
plt.title(str(distance))

boxes = []
classes = []

for box in root.findall('.//bndbox'):
    xmin = int(box.find('xmin').text)
    ymin = int(box.find('ymin').text)
    xmax = int(box.find('xmax').text)
    ymax = int(box.find('ymax').text)
    coors = [xmin, ymin, xmax, ymax]
    boxes.append(coors)

for obj in root.findall('.//object'):
    objClass = int(obj.find('name').text)
    classes.append(objClass)

print(boxes)
print(classes)
print(distance)
ax = plt.gca()



for i,box in enumerate(boxes):
    x1, y1, x2, y2 = box
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    # centers.append([cx,cy])
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    class_name = ''
    if classes[i] ==0:
        class_name = 'bear'
    elif classes[i] ==1:
        class_name = 'bile'
    elif classes[i] ==2:
        class_name = 'camel'
    elif classes[i] ==3:
        class_name = 'car'
    elif classes[i] ==4:
        class_name = 'helicopter'
    elif classes[i] ==5:
        class_name = 'ladder'
    elif classes[i] ==6:
        class_name = 'sofa'
    ax.annotate(class_name, (x1+4, y1-10), color='red', fontsize=10, ha='center', va='center')
    ax.add_patch(rect)






# for center in centers:
# 	x, y = center
# 	cir = Circle((x, y), 3, fill=True, color='red')
# 	ax.add_patch(cir)

plt.show()