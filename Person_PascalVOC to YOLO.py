#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xml.etree.ElementTree as ET


# In[2]:


def convert_voc_to_yolo(voc_dir, yolo_dir, class_list):
    os.makedirs(yolo_dir, exist_ok=True)
    class_dict = {cls: i for i, cls in enumerate(class_list)}

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_file = os.path.join(yolo_dir, xml_file.replace(".xml", ".txt"))
        with open(yolo_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in class_dict:
                    continue  

                class_id = class_dict[class_name]
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


# In[3]:


voc_dir = "C:/Users/shaha/Downloads/syook/datasets/labels"
yolo_dir = "C:/Users/shaha/Downloads/syook/datasets/yolo_labels"
class_file = "C:/Users/shaha/Downloads/syook/datasets/person_classes.txt"


# In[4]:


with open(class_file, "r") as f:
    class_list = [line.strip() for line in f.readlines()]


# In[5]:


convert_voc_to_yolo(voc_dir, yolo_dir, class_list)
print("Conversion completed successfully!")


# In[ ]:




