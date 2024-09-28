from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
from PIL import Image
from PIL import ImageDraw




def Generate (dlow, dhigh, dlabel):

    objects = ["ballon", "bike", "camel", "car", "helicopter", "ladder", "sofa"]


    for i in tqdm(range(dlow,dhigh)):

        file_id = f'{i:05}'

        distance = np.random.randint(low = dlow + 20, high = dhigh -20)
        distance_label = dlabel

        object1 = np.random.randint(low = 0, high=7)
        object2 = np.random.randint(low = 0, high=7)
        image1 = Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/" + objects[object1] + ".png")
        image2 = Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/" + objects[object2] + ".png")

        # size_dif1 = np.random.randint(low = 0, high=50)
        # size_dif2 = np.random.randint(low = 0, high=50)
        size_dif1 = 0
        size_dif2 = 0
        
        object1_size = (50 + size_dif1, 50 + size_dif1)
        object2_size = (50 + size_dif2, 50 + size_dif2)

        image1 = image1.resize(object1_size)
        image2 = image2.resize(object2_size)

        x1_start  = np.random.randint(low = 1 + 50 + size_dif1 , high = 480 - (50 + size_dif1 + int(distance / np.sqrt(2)) + 50 + size_dif2 ) )
        y1_start  = np.random.randint(low = 1 + 50 + size_dif1 , high = 480 - (50 + size_dif1 + int(distance / np.sqrt(2)) + 50 + size_dif2 ) )
        x1_end    = x1_start + 50 + size_dif1
        y1_end    = y1_start + 50 + size_dif1
        x1_center = int((x1_start + x1_end) / 2)
        y1_center = int((y1_start + y1_end) / 2)

        x2_center = x1_center + int(distance / np.sqrt(2))
        y2_center = y1_center + int(distance / np.sqrt(2))
        x2_start  = x2_center - int((50 + size_dif2)/2) 
        y2_start  = y2_center - int((50 + size_dif2)/2)
        x2_end    = x2_center + int((50 + size_dif2)/2) 
        y2_end    = y2_center + int((50 + size_dif2)/2)
        
        ActualDistance=np.sqrt((y2_center-y1_center)**2 + (x2_center-x1_center)**2)

        canvas= Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/canvas.jpg")
        

        canvas.paste(image1, (x1_start, y1_start), image1)
        canvas.paste(image2, (x2_start, y2_start), image2)

        # I1 = ImageDraw.Draw(canvas)
        # I1.ellipse(xy = (x1_center, y1_center, x1_center+5, y1_center+5), 
        #         fill = (0, 255, 0))
        # I1.ellipse(xy = (x2_center, y2_center, x2_center+5, y2_center+5), 
        #         fill = (255, 0, 0))
        # I1.text((10, 10), str(dis), fill=(255, 0, 0))
        # I1.text((10, 40), str(distance), fill=( 0,255, 0))

        canvas.save("/home/el/Documents/My Object Dataset/Dataset/images/" + str(file_id) + ".jpg", quality=100)
        
        #Annotate
        root = ET.Element('annotation')

        folder = ET.SubElement(root, 'folder')
        folder.text = 'Dataset'

        filename = ET.SubElement(root, 'filename')
        filename.text = str(file_id) + ".jpg"

        path = ET.SubElement(root, 'path')
        path.text = "/home/el/Documents/My Object Dataset/Dataset/images/" + str(file_id) + ".jpg"

        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Objects Dataset (Generated)'

        size = ET.SubElement(root, 'size')
        wi = ET.SubElement(size, 'width')
        wi.text = str(480)
        he = ET.SubElement(size, 'height')
        he.text = str(480)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)

        segmented = ET.SubElement(root, 'segmented')
        segmented.text=str(0)

        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = str(object1)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(x1_start)
        ymin.text = str(y1_start)
        xmax.text = str(x1_end)
        ymax.text = str(y1_end)

        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = str(object2)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(x2_start)
        ymin.text = str(y2_start)
        xmax.text = str(x2_end)
        ymax.text = str(y2_end)

        objdist = ET.SubElement(root, 'distance')
        objdist.text = str(ActualDistance)

        distanceLabel = ET.SubElement(root, 'distanceLabel')
        distanceLabel.text = str(distance_label)

        treestr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open("/home/el/Documents/My Object Dataset/Dataset/annots/"+str(file_id)+'.xml', "w") as f:
            f.write(treestr)


def Generate0 (dlow, dhigh, dlabel):

    objects = ["ballon", "bike", "camel", "car", "helicopter", "ladder", "sofa"]


    for i in tqdm(range(dlow,dhigh)):

        file_id = f'{i:05}'

        distance = np.random.randint(low = dlow + 60, high = dhigh -20)
        distance_label = dlabel

        object1 = np.random.randint(low = 0, high=7)
        object2 = np.random.randint(low = 0, high=7)
        image1 = Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/" + objects[object1] + ".png")
        image2 = Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/" + objects[object2] + ".png")

        # size_dif1 = np.random.randint(low = 0, high=50)
        # size_dif2 = np.random.randint(low = 0, high=50)
        size_dif1 = 0
        size_dif2 = 0
        
        object1_size = (60 + size_dif1, 60 + size_dif1)
        object2_size = (60 + size_dif2, 60 + size_dif2)

        image1 = image1.resize(object1_size)
        image2 = image2.resize(object2_size)

        x1_start  = np.random.randint(low = 1 + 60 + size_dif1 , high = 480 - (60 + size_dif1 + int(distance / np.sqrt(2)) + 60 + size_dif2 ) )
        y1_start  = np.random.randint(low = 1 + 60 + size_dif1 , high = 480 - (60 + size_dif1 + int(distance / np.sqrt(2)) + 60 + size_dif2 ) )
        x1_end    = x1_start + 60 + size_dif1
        y1_end    = y1_start + 60 + size_dif1
        x1_center = int((x1_start + x1_end) / 2)
        y1_center = int((y1_start + y1_end) / 2)

        x2_center = x1_center + int(distance / np.sqrt(2))
        y2_center = y1_center + int(distance / np.sqrt(2))
        x2_start  = x2_center - int((60 + size_dif2)/2) 
        y2_start  = y2_center - int((60 + size_dif2)/2)
        x2_end    = x2_center + int((60 + size_dif2)/2) 
        y2_end    = y2_center + int((60 + size_dif2)/2)
        
        ActualDistance=np.sqrt((y2_center-y1_center)**2 + (x2_center-x1_center)**2)

        canvas= Image.open("/home/el/Documents/My Object Dataset/SourceCode/images/canvas.jpg")
        

        canvas.paste(image1, (x1_start, y1_start), image1)
        canvas.paste(image2, (x2_start, y2_start), image2)

        # I1 = ImageDraw.Draw(canvas)
        # I1.ellipse(xy = (x1_center, y1_center, x1_center+5, y1_center+5), 
        #         fill = (0, 255, 0))
        # I1.ellipse(xy = (x2_center, y2_center, x2_center+5, y2_center+5), 
        #         fill = (255, 0, 0))
        # I1.text((10, 10), str(dis), fill=(255, 0, 0))
        # I1.text((10, 40), str(distance), fill=( 0,255, 0))

        canvas.save("/home/el/Documents/My Object Dataset/Dataset/images/" + str(file_id) + ".jpg", quality=100)
        
        #Annotate
        root = ET.Element('annotation')

        folder = ET.SubElement(root, 'folder')
        folder.text = 'Dataset'

        filename = ET.SubElement(root, 'filename')
        filename.text = str(file_id) + ".jpg"

        path = ET.SubElement(root, 'path')
        path.text = "/home/el/Documents/My Object Dataset/Dataset/images/" + str(file_id) + ".jpg"

        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Objects Dataset (Generated)'

        size = ET.SubElement(root, 'size')
        wi = ET.SubElement(size, 'width')
        wi.text = str(480)
        he = ET.SubElement(size, 'height')
        he.text = str(480)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(3)

        segmented = ET.SubElement(root, 'segmented')
        segmented.text=str(0)

        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = str(object1)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(x1_start)
        ymin.text = str(y1_start)
        xmax.text = str(x1_end)
        ymax.text = str(y1_end)

        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = str(object2)
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(x2_start)
        ymin.text = str(y2_start)
        xmax.text = str(x2_end)
        ymax.text = str(y2_end)

        objdist = ET.SubElement(root, 'distance')
        objdist.text = str(ActualDistance)

        distanceLabel = ET.SubElement(root, 'distanceLabel')
        distanceLabel.text = str(distance_label)

        treestr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open("/home/el/Documents/My Object Dataset/Dataset/annots/"+str(file_id)+'.xml', "w") as f:
            f.write(treestr)



# 0 <= Distance < 120 ; Very close ; Label = 0
Generate0 (0, 120, 0)

# 120 <= Distance < 240 ;  Close; Label = 1
Generate (120, 240, 1)

# 240 <= Distance < 360 ; Far; Label = 2
Generate (240, 360, 2)

# 360 <= Distance < 480 ; Too Far; Label = 3
Generate (360, 480, 3)







