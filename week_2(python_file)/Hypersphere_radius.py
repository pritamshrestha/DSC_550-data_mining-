#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name:Pritam Shrestha
# FileName:Hypersphere_Radius
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:Hypersphere radius with d-dimentions
# Due Date:06/20/20
# Assignment No:2.1
# Reference:http://www.dataminingbook.info/uploads/videos/lecture7/


# In[3]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
# function to calculate radious of hyperSphere 
# given volume is 1
list_redious=[]
def volume(d): 
    for i in d:
        v=1
        r=gamma((i/2+1)**1/i)/pi**0.5*v**1/i
        list_redious.append(r)
# dimension of the hypersphere
d=[x for x in range(1,101)]
       
volume(d)
print(list_redious)
plt.plot(d,list_redious)
plt.xlabel("d")
plt.ylabel("Volume")
plt.show()


# In[ ]:


# Thanks!!!

