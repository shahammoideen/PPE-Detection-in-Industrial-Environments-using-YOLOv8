#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO


# In[2]:


PERSON_MODEL_PATH = "C:/Users/shaha/runs/detect/yolo_person2/weights/best.onnx"
PPE_MODEL_PATH = "C:/Users/shaha/runs/detect/yolo_ppe4/weights/best.onnx"


# In[3]:


INPUT_DIR = "C:/Users/shaha/Downloads/syook/datasets/test_image"
OUTPUT_DIR = "C:/Users/shaha/Downloads/syook/datasets/inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[4]:


person_model = YOLO(PERSON_MODEL_PATH, task="detect")
ppe_model = YOLO(PPE_MODEL_PATH, task="detect")


# In[5]:


PPE_CLASSES = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]


# In[6]:


def run_inference():
    for img_name in os.listdir(INPUT_DIR):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue  

        img_path = os.path.join(INPUT_DIR, img_name)
        image = cv2.imread(img_path)

        # Run person detection
        person_results = person_model(img_path)
        person_bboxes = []
        
        if image is None:
            print(f"Skipping {img_name} (Error loading image)")
            continue

        person_results = person_model(image)
        person_bboxes = person_results[0].boxes.xyxy.cpu().numpy() if person_results else []

        print(f"Image: {img_name} - Detected {len(person_bboxes)} persons")

        # Draw person bounding boxes
        for bbox in person_bboxes:
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue for persons
            cv2.putText(image, "Person", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Crop person region
            person_crop = image[ymin:ymax, xmin:xmax]
            if person_crop.size == 0:
                print(f"Skipping empty crop for {img_name}")
                continue

            # Convert BGR to RGB for PPE detection (if model expects RGB)
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Lower confidence threshold for PPE detection
            ppe_results = ppe_model(person_crop_rgb, conf=0.1)

            # Debug: Check if any PPE detected
            if len(ppe_results[0].boxes) == 0:
                print(f"No PPE detected in cropped image from {img_name}")
            else:
                # Draw PPE bounding boxes
                for result in ppe_results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        px_min, py_min, px_max, py_max = map(int, box.cpu().numpy())
                        class_id = int(cls.cpu().numpy())
                        confidence = float(conf.cpu().numpy())

                        # Convert PPE box coordinates to original image coordinates
                        abs_xmin = xmin + px_min
                        abs_ymin = ymin + py_min
                        abs_xmax = xmin + px_max
                        abs_ymax = ymin + py_max

                        cv2.rectangle(image, (abs_xmin, abs_ymin), (abs_xmax, abs_ymax), (0, 255, 0), 2)
                        label = f"{PPE_CLASSES[class_id]}: {confidence:.2f}"
                        cv2.putText(image, label, (abs_xmin, abs_ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save processed image
        output_img_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(output_img_path, image)
        print(f"Processed: {img_name} -> Saved at {output_img_path}")


# In[7]:


if __name__ == "__main__":
    run_inference()
    print("Inference completed!")


# In[ ]:




