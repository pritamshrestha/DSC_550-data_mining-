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


# In[3]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
## Function to calculate Volume of hyperSphere where radius is 1 due to unit radious.
# creating empty list for volume
list_vol=[]

# calculating the volume of the hypersphere with range(1,50) dimensions
d=[x for x in range(1,50)]  
def volume(d): 
    for i in d:
        r=1
        # applying formula to calculate volume with unit radius
        vol=(pi**i/2/gamma(i/2+1))*r**i
        list_vol.append(vol)

def plot(d,list_vol):
    plt.plot(d,list_vol)
    plt.xlabel("d")
    plt.ylabel("volume")
    plt.show()

if __name__=='__main__':
    #calling function inside the try block to catch the errors
    try:
        volume(d)
        plot(d,list_vol)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally block is executed whether exception is handled or not!!")
    
             


# In[ ]:


# The volume of the hypersphere is increasing with high dimension and after around dimension 20 it is decreasing to Zero.

