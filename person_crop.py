#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
from ultralytics import YOLO


# In[2]:


def crop_persons(model_path, input_images_path, output_cropped_path):
    model = YOLO(model_path)  
    os.makedirs(output_cropped_path, exist_ok=True)

    for img_file in os.listdir(input_images_path):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(input_images_path, img_file)
            image = cv2.imread(img_path)
            results = model(img_path)

            for i, box in enumerate(results[0].boxes.xyxy):
                x_min, y_min, x_max, y_max = map(int, box)
                cropped_person = image[y_min:y_max, x_min:x_max]

                cropped_filename = f"{os.path.splitext(img_file)[0]}_crop{i}.jpg"
                cv2.imwrite(os.path.join(output_cropped_path, cropped_filename), cropped_person)

    print(f"Person cropping completed for {input_images_path}!")


# In[3]:


person_model_path = "C:/Users/shaha/runs/detect/yolo_person2/weights/best.pt"
train_images_path = "C:/Users/shaha/Downloads/syook/datasets/images/train"
val_images_path = "C:/Users/shaha/Downloads/syook/datasets/images/val"
cropped_train_path = "C:/Users/shaha/Downloads/syook/cropped_persons/train"
cropped_val_path = "C:/Users/shaha/Downloads/syook/cropped_persons/val"


# In[4]:


crop_persons(person_model_path, train_images_path, cropped_train_path)
crop_persons(person_model_path, val_images_path, cropped_val_path)
print("Person cropping completed successfully for both train and val datasets!")

