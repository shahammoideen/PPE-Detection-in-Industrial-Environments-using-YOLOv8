#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import xml.etree.ElementTree as ET


# In[2]:


def convert_ppe_voc_to_yolo(voc_dir, cropped_images_dir, yolo_dir, class_list):
    os.makedirs(yolo_dir, exist_ok=True)
    class_dict = {cls: i for i, cls in enumerate(class_list)}

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        base_filename = os.path.splitext(xml_file)[0]

        
        for cropped_img in os.listdir(cropped_images_dir):
            if base_filename in cropped_img:
                cropped_yolo_file = os.path.join(yolo_dir, cropped_img.replace(".jpg", ".txt"))

                with open(cropped_yolo_file, "w") as f:
                    for obj in root.findall("object"):
                        class_name = obj.find("name").text
                        if class_name not in class_dict:
                            continue  

                        class_id = class_dict[class_name]
                        bbox = obj.find("bndbox")
                        orig_xmin = int(bbox.find("xmin").text)
                        orig_ymin = int(bbox.find("ymin").text)
                        orig_xmax = int(bbox.find("xmax").text)
                        orig_ymax = int(bbox.find("ymax").text)

                        
                        x_center = (orig_xmin + orig_xmax) / 2.0 / img_width
                        y_center = (orig_ymin + orig_ymax) / 2.0 / img_height
                        bbox_width = (orig_xmax - orig_xmin) / img_width
                        bbox_height = (orig_ymax - orig_ymin) / img_height

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print(f"PPE annotation conversion completed for {voc_dir}!")


# In[3]:


voc_annotations_path = "C:/Users/shaha/Downloads/syook/datasets/labels"
cropped_images_train = "C:/Users/shaha/Downloads/syook/cropped_persons/train"
cropped_images_val = "C:/Users/shaha/Downloads/syook/cropped_persons/val"
yolo_labels_train = "C:/Users/shaha/Downloads/syook/datasets/labels_ppe/train"
yolo_labels_val = "C:/Users/shaha/Downloads/syook/datasets/labels_ppe/val"
class_file = "C:/Users/shaha/Downloads/syook/datasets/ppe_classes.txt"


# In[4]:


with open(class_file, "r") as f:
    class_list = [line.strip() for line in f.readlines()]


# In[5]:


convert_ppe_voc_to_yolo(voc_annotations_path, cropped_images_train, yolo_labels_train, class_list)
convert_ppe_voc_to_yolo(voc_annotations_path, cropped_images_val, yolo_labels_val, class_list)
print("PPE annotations successfully converted for both train and val datasets!")


# In[ ]:




