#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import random


# In[2]:


image_dir = "C:/Users/shaha/Downloads/syook/datasets/images"  
label_dir = "C:/Users/shaha/Downloads/syook/datasets/yolo_labels"  


# In[3]:


train_image_dir = "C:/Users/shaha/Downloads/syook/datasets/images/train"
val_image_dir = "C:/Users/shaha/Downloads/syook/datasets/images/val"
train_label_dir = "C:/Users/shaha/Downloads/syook/datasets/labels/train"
val_label_dir = "C:/Users/shaha/Downloads/syook/datasets/labels/val"


# In[4]:


os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)


# In[5]:


image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)


# In[6]:


train_ratio = 0.8
train_count = int(len(image_files) * train_ratio)
train_images = image_files[:train_count]
val_images = image_files[train_count:]


# In[7]:


for img in train_images:
    src_image = os.path.join(image_dir, img)
    dest_image = os.path.join(train_image_dir, img)
    shutil.move(src_image, dest_image)

    label_file = img.rsplit(".", 1)[0] + ".txt"  # Change file extension to .txt
    src_label = os.path.join(label_dir, label_file)
    dest_label = os.path.join(train_label_dir, label_file)

    if os.path.exists(src_label):
        shutil.move(src_label, dest_label)


# In[8]:


for img in val_images:
    src_image = os.path.join(image_dir, img)
    dest_image = os.path.join(val_image_dir, img)
    shutil.move(src_image, dest_image)

    label_file = img.rsplit(".", 1)[0] + ".txt"  # Change file extension to .txt
    src_label = os.path.join(label_dir, label_file)
    dest_label = os.path.join(val_label_dir, label_file)

    if os.path.exists(src_label):
        shutil.move(src_label, dest_label)

print("Dataset split completed successfully!")

