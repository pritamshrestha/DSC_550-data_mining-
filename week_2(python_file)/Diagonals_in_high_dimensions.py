#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name:Pritam Shrestha
# FileName:Diagonals_in_high_dimensions
# Date:6/19/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:angle between orthogonal and density of the multivariate normal
# Due Date:06/20/20
# Assignment No:2.1


# In[ ]:


"""
Theory
mue=0
sigma=1
then density of points around the mean
f(x)=(1/sqrt(2*pi)**d)*e**-XT
"""


# Your goal is the compute the empirical probability mass function (EPMF) for the random variable X that represents the angle (in degrees) between any two diagonals in high dimensions.
# 
# Assume that there are d primary dimensions (the standard axes in cartesian coordinates), with each of them ranging from -1 to 1. There are 2d additional half-diagonals in this space, one for each corner of the d-dimensional hypercube.
# 
# Write a script that randomly generates n=100000 pairs of half-diagonals in the d-dimensional hypercube, and computes the angle between them (in degrees).
# 
# Plot the EPMF for three different values of d, as follows d=10,100,1000. What is the min, max, value range, mean and variance of X for each value of d?
# 
# What would you have expected to have happened analytically? In other words, derive formulas for what should happen to angle between half-diagonals as d→∞. Does the EPMF conform to this trend? Explain why? or why not?
# 
# What is the expected number of occurrences of a given angle θ between two half-diagonals, as a function of d (the dimensionality) and n (the sample size)?

# In[48]:


import numpy as np
import math
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
# creating class counter for PMF calculation.
class EPmf(Counter):
# normalizing pmf to add probabalities by 1
    def Normalize_pmf(self):
       
        sum_ = float(sum(self.values()))
        for key in self:
            self[key] /=sum_
# adding two distribution
    def add_with_other(self, other):
        
        epmf = EPmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                epmf[key1 + key2] += prob1 * prob2
        return epmf

    def hash_value(self):
        """Returns an integer hash value."""
        return id(self)

    def other(self, other):
        return self is other

    def render_value(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))


def find_angle(point1, point2):
   # applying formula to calculate angle
    return np.dot(point1, point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))

#Function to get the array of angles for d-dimensions

def get_angle_in_d(n, d):
    results = np.zeros(n)
    i = 0
    while(i < n):
        # generating 2d array
        points_pre = np.random.rand(2, d)   
        points_pre[points_pre <= 0.5] = -1
        points_pre[points_pre > 0.5] = 1
        # angle between two points
        cos_theta = find_angle(points_pre[0], points_pre[1]) 
        # converting angle to degree
        results[i] = round(math.degrees(math.acos(cos_theta)), 2)  
        i = i+1
    return results

# creating function for plot
def plot_epmf(angle):
   
    p = EPmf(angle)
    p.Normalize_pmf()
    ky = []
    pr = []
    for key, prob in p.items():
        ky.append(key)
        pr.append(prob)

    plt.hist(pr, bins=20)
    plt.xlabel('Size')
    plt.ylabel('EPMF')
    plt.show()


if __name__ == '__main__':
# given no of half diagonals pair
    n = 100000                                      

    # Calculating angle between two half-diagonals for d=10
    d = 10
    angle = get_angle_in_d(n, d)                          
    plot_epmf(angle)    
    #summary of angle
    summary=stats.describe(angle)   
    print('summary for d = 10 :\n',summary, '\n')  

    # Calculating angle between two half-diagonals for d=10
    d = 100
    angle = get_angle_in_d(n, d)                          
    plot_epmf(angle)  
    #summary of angle
    summary=stats.describe(angle)   
    print('summary for d = 100 :\n', summary, '\n') 

    # Calculating angle between two half-diagonals for d=10
    d = 1000
    angle = get_angle_in_d(n, d)                        
    plot_epmf(angle)
    #summary of angle
    summary=stats.describe(angle)   

    print('Summary for d=1000\n', summary, '\n')

    print('with increase in dimensions angle becomes 90 degree.That implies that in high dimentions all of the digonal'
          'vectors are perpendicular to all the cordinates axes because there are 2**d corners in a d-dimensional hypercubes'
          ',there are 2**d diagonal vectors from the origin to each of the corners.')

