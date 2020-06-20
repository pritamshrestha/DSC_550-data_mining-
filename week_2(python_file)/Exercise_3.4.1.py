#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name:Pritam Shrestha
# FileName:Exercise_4.3.1
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:S-Curve
# Due Date:06/20/20
# Assignment No:2.1
# Reference:https://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/


# In[15]:


# question:
# Evaluate the S-Curve 1-(1-s**r)**b for s=01,0.2.....0.9 for the following values of r and b:
# r=3 and b=10
# r=6 and b=20
# r=5 and b=50


# In[2]:


### import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

# Given equation
#e=1-(1-s**r)**b
# given values of s as list is:
#s_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Given values of r ans b

r1 = 3
b1 = 10
r2 = 6
b2 = 20
r3 = 5
b3 = 50
s_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def compute_curve(row, band):
    s_curve = []
    for i in range(0,len(s_values)):
        s_curve.append(1-(1-s_values[i]**row)**band)
    return s_curve
def plot_curve(s_curve_list):
    x=s_values
    y=s_curve_list
    plt.plot(x,y)
    plt.xlabel('S_Values')
    plt.ylabel("S_Curve_Values")
    plt.title("S_Value Vs S_Curve Plot")
    

if __name__ == '__main__':
     # try block for execution
    try:
        s_curve_1=compute_curve(r1,b1)
        
        
        s_curve_2=compute_curve(r2,b2)
        
        s_curve_3=compute_curve(r3,b3)
        # plotting curve with different s_curve values
        plot_curve(s_curve_1)
       
        plot_curve(s_curve_2)
       
        plot_curve(s_curve_3)
       
    # exception block to catch any exceptions during execution
    except Exception as exception:
        print('exception')
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally, the block is executed whether an exception is handled or not!!")    


    


# In[55]:


# Thanks!!!

