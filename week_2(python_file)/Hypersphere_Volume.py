#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name:Pritam Shrestha
# FileName:Hypersphere_volume
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:Hypersphere volume with d-dimentions
# Due Date:06/20/20
# Assignment No:2.1
# Reference:http://www.dataminingbook.info/uploads/videos/lecture7/


# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
## Function to calculate Volume of hyperSphere where radius is 1 due to unit radious.
# creating empty list for volume
list_vol=[]
def volume(d): 
    for i in d:
        r=1
        # applying formula to calculate volume with unit radius
        vol=(pi**i/2/gamma(i/2+1))*r**i
        list_vol.append(vol)
# calculating the volume of the hypersphere with range(1,50) dimensions
d=[x for x in range(1,50)]      
volume(d)
print(list_vol)
# plotting using matplotlib
plt.plot(d,list_vol)
plt.xlabel("d")
plt.ylabel("Volume")
plt.show()


# In[ ]:


# the volume of the hypersphere is increasing with high dimentions and after around dimention 25 it is decreasing to Zero.

