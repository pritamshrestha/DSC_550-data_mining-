#!/usr/bin/env python
# coding: utf-8

# For the three clusters of fig.7.8:
# a) compute the representation of the cluster as in the BFR Algorithm.That is,compute N,SUM,and SUMSQ
# b) Compute the variance and standard deviation of each cluster in each of the two dimentions.

# In[53]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Exercise No:7.3.4
# Reference:Mining of massive dataset

import pandas as pd
import numpy as np
import traceback
import math


# In[54]:


# solution of a
# there is three clusters
# for cluster one
x1=[2,5,3]
y1=[2,2,4]
# for cluster two
x2=[9,12,11,10,12]
y2=[3,3,4,5,6,]
# for cluster three
x3=[4,6,7,4]
y3=[8,8,10,10]


# defining function to calculate n
def compute_n(a):
    n=len(a)
    return n
    
# defining function to calculate sum 
def SUM(x,y):
    total=[]
    x=sum(x)
    y=sum(y)
    total.append(x)
    total.append(y)
    return total

# defining function to calculate SUMSQ
def SUMSQ(x,y):
    total=[]
    x_list=[]
    for i in x:
        x_list.append(i*i)
    y_list=[]
    for i in y:
        y_list.append(i*i)
    a=sum(x_list)
    b=sum(y_list)
    total.append(a)
    total.append(b)
    return total
# calling from the main function
if __name__=='__main__':
    try:
        print("For cluster One")
        print("Length of the given points is:",compute_n(x1))
        print("Sum of given no of points is:",SUM(x1,y1))
        print("Sum of SUMSQ is:",SUMSQ(x1,y1))
        print("For cluster two")
        print("Length of the given points is:",compute_n(x2))
        print("Sum of given no of points is:",SUM(x2,y2))
        print("Sum of SUMSQ is:",SUMSQ(x2,y2))
        print("For cluster Three")
        print("Length of the given points is:",compute_n(x3))
        print("Sum of given no of points is:",SUM(x3,y3))
        print("Sum of SUMSQ is:",SUMSQ(x3,y3))
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally block is executed wheather exception is handled or not!!")
        
        


# In[62]:


# solution of b
SUMSQ1=[28,24]
SUMSQ2=[590,95]
SUMSQ3=[117,328]
N1=3
N2=5
N3=4
SUM1=[10,8]
SUM2=[54,21]
SUM3=[21,36]
# in the second dimention 
def variance(sumsqi,sumi,N):
    a=sumsqi/N
    b=(sumi/N)**2
    variance=a-b
    st_d=(variance)**1/2
    print("Variance is:",variance)
    print("Standard deviation is:",st_d)
# main function

if __name__=='__main__':
    
    try:
        print("variance and sandard deviation of cluster 1")    
        variance(24,8,3)  
        print("variance and sandard deviation of cluster 2")  
        variance(95,21,5)
        print("variance and sandard deviation of cluster 3")  
        variance(328,36,4)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")

