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


# In[6]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
# function to calculate radious of hyperSphere 
# given volume is 1
list_radius=[]
# dimensions list
d=[x for x in range(1,100)]
def radius(d): 
    
    for i in d:
        v=1
        r=gamma((i/2+1)**1/i)/pi**0.5*v**1/i
        list_radius.append(r)

def plot(d,list_radius):
    plt.plot(d,list_radius)
    plt.xlabel("d")
    plt.ylabel("radius")
    plt.show()

if __name__=='__main__':
    #calling function inside the try block to catch the errors
    try:
        radius(d)
        plot(d,list_radius)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally block is executed whether exception is handled or not!!")
    
         


# In[ ]:


# Thanks!!!

