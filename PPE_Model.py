#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO


# In[2]:


model = YOLO("yolov8n.pt")


# In[3]:


model.train(data="C:/Users/shaha/Downloads/syook/ppe_dataset.yaml",
            epochs=50, batch=16, imgsz=640, name="yolo_ppe")


# In[4]:


model.export(format="onnx")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




