#!/usr/bin/env python
# coding: utf-8

# In[1]:


yaml_content = """path: C:/Users/shaha/Downloads/syook/datasets
train: C:/Users/shaha/Downloads/syook/datasets/images/train/
val: C:/Users/shaha/Downloads/syook/datasets/images/val/

nc: 1

names:
  0: person
"""


# In[2]:


yaml_path = "C:/Users/shaha/Downloads/syook/datasets/person_dataset.yaml"


# In[3]:


with open(yaml_path, "w") as file:
    file.write(yaml_content)


# In[4]:


print(f"person_dataset.yaml file created successfully at: {yaml_path}")

