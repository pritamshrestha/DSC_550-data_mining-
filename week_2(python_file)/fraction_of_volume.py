#!/usr/bin/env python
# coding: utf-8

# In[75]:


# Name:Pritam Shrestha
# FileName:fraction_of_volume
# Date:6/19/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:fraction of volume with d-dimentions
# Due Date:06/20/20
# Assignment No:2.1


# Fraction of Volume: Assume we have a hypercube of edge length l=2 centered at the origin (0,0,⋯,0). Generate n=10,000 points uniformly at random for increasing 
# dimensionality d=1,⋯,100. Now answer the following questions:
# 
# Plot the fraction of points that lie inside the largest hypersphere that can be inscribed 
# inside the hypercube with increasing d. After how many dimensions does the fraction go to essentially zero?
# 

# In[76]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from math import pi
"""
volume_of_hypersphere(V(sd(r)))=((pi**d/2)/gamma(d/2+1))*r**d
volume_of_hypercube(V(hd(l)))=l**d
where l=2r
so, V(hd(2r))=(2r)**d
Now ratio of fraction of volume=(V(sd(r)))/(V(hd(l)))=(pi**d)/2**d*gamma(d/2+1)
"""
# creating global variable 
fraction_of_point=[]
d=[x for x in range(1,100)] 
def calculate_ratio(d):  
    for i in d:
        l=2
        r=l/2
        # if r=1 then the formula of the volume of the sphere is (pi**i/2)/gamma(i/2+1)
        volume=(pi**i/2)/gamma(i/2+1)
        fraction_of_point.append(volume)

# function for plot
def plot(d,fraction_of_point):
    plt.plot(d, fraction_of_point)
    plt.xlabel("d")
    plt.ylabel("Fraction Of point")
    plt.show()

if __name__=='__main__':
    #calling function inside the try block to catch the errors
    try:
        calculate_ratio(d)
        plot(d,fraction_of_point)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed whether exception is handled or not!!")
    
    


# In[77]:


# hence, a value of fraction of volume goes to Zero after around dimention-38.


# Plot the fraction of points in the thin shell of width ϵ=0.01 inside the hypercube (i.e., the difference between the outer hypercube and inner hypercube, or the thin shell along the boundaries). What is the trend that you see? After how many dimensions does the fraction of volume in the thin shell go to 100% (use binary search or increase the dimensionality in steps of 10 to answer this. You may use maximum dimensions of up to 2000, and you may use a threshold of 0.0001 to count the volume as essentially being 1 in the shell, i.e., a volume of 0.9999 can be taken to be equal to 1 for finding the smallest dimension at which this happens).

# In[78]:


# solution
"""
volume of the thin shell in d dimention==1(100%)
formula
vol(sd(r,e))=1-(1-e/r)**d
where e=0.0001
"""
# first made global variable as list
fraction_of_point_of_thin_sphere=[]
d=[x for x in range(1,400)] 
def calculate_volume_of_thin_shell(d): 
    for i in d:   
        l=2
        e=0.01
        r=l/2
        # with the higher dimention it goes to 1
        volume=1-(1-e/r)**i
        fraction_of_point_of_thin_sphere.append(volume)  
        
        
 # creating function to plot       
def plot(d,fraction_of_point_of_thin_sphere):
    plt.plot(d, fraction_of_point_of_thin_sphere)
    plt.xlabel("d")
    plt.ylabel("Fraction Of point of thin shell")
    plt.show()
print(fraction_of_point_of_thin_sphere)

if __name__=='__main__':
    # calling function inside try blck to catch the eror
    try:
        calculate_volume_of_thin_shell(d)
        plot(d,fraction_of_point_of_thin_sphere)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed whether exception is handled or not!!")
    


# In[79]:


# we can see in the plot when we increase the dimension, all points go to the boundary after around 330 dimention.

