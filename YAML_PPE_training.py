#!/usr/bin/env python
# coding: utf-8

# In[1]:


yaml_content = """path: C:/Users/shaha/Downloads/syook/datasets
train: C:/Users/shaha/Downloads/syook/cropped_persons/train
val: C:/Users/shaha/Downloads/syook/cropped_persons/val
nc: 9 
names:
  0: hard-hat
  1: gloves
  2: mask
  3: glasses
  4: boots
  5: vest
  6: ppe-suit
  7: ear-protector
  8: safety-harness

"""


# In[2]:


yaml_path = "C:/Users/shaha/Downloads/syook/ppe_dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)


# In[3]:


print(f"PPE dataset YAML file created successfully at {yaml_path}")

