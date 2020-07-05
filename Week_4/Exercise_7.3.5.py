#!/usr/bin/env python
# coding: utf-8

# 7.3.5)Suppose a cluster of three-dimantional points has standard deviation of 2,3 and 5 in the three dimentions, in that order.
# Compute the Mahalanobis distance between the origin(0,0,0) and the point (1,-3,4)

# In[2]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Exercise No:7.3.5
# Reference:Mining of massive dataset


# In[3]:


import traceback
# defining function to calculate euclidian distance
def calculate_euclidian_dist(p1,p2,s):
    distance=0
    for i in range(len(p1)):
        for m in range(len(s)):
            distance+=((p1[i]-p2[i])/s[m])**2
        return distance**0.5
        return distance
    

# given points and standard deviations
# origin
p1=(0,0,0)
# another point 
p2=(1,-3,4)
# given list of standard deviations
s=[2,3,5]
if __name__=='__main__':
    # calling all function to calculate euclidian distance of each cluster.
    try:
        d=calculate_euclidian_dist(p1,p2,s)
        print("The Mahalanobis distance is:",d)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")

