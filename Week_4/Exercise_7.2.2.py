#!/usr/bin/env python
# coding: utf-8

#  Exercise 7.2.2 How would the clustering of example 7.2 change if we used for the distance between two clusters:
#     a) The minimum of the distances between any two points, one from each cluster.
#     b) The average of the distance between pairs of points,one from each of the two clusters.

# In[51]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Exercise No:7.2.2
# Reference:Mining of massive dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import scipy.cluster.hierarchy as sch
# given corodinates of the points
x_point_list=[2,5,3,9,12,11,10,12,6,4,7,4]
y_point_list=[2,2,4,3,3,4,5,6,8,8,10,10]

# creating function to scatter plot
def plot(x,y):
    plt.scatter(x,y)
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.show()
plot(x_point_list,y_point_list)


# In[52]:



import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# checking dendrogram to see clustering
df={'x_values':[2,5,3,9,12,11,10,12,6,4,7,4],
     'y_values':[2,2,4,3,3,4,5,6,8,8,10,10]}
data=pd.DataFrame(df)

x=data.iloc[:,[0,1]].values
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))


# In[53]:


hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='Cluster3')


# In[54]:


# from above two plots, it is clear that there are three clusters.
# now lets calculate the minimum distance between any poins, one from each cluster.
# now create the function to calculate euclidian distance 
def calculate_euclidian_dist(p1,p2):
    distance=0
    for i in range(len(p1)):
        distance+=p1[i]-p2[i]**2
    return distance**0.5
    return distance

def cluster_one():
    z=[]
    a=[2,3]
    b=[5,2]
    c=[3,4]
    m=(calculate_euclidian_dist(a,b))
    n=(calculate_euclidian_dist(a,c))
    o=(calculate_euclidian_dist(c,b))
    z.append(m)
    z.append(n)
    z.append(o)
    p = sorted(z, key=lambda x: x.imag) 
    print("Minimum distance of cluster 1 is:",p[0])


def cluster_two():
    two_list=[]
    d=[4,8]
    e=[6,8]
    f=[7,10]
    g=[4,10]
    two_a=(calculate_euclidian_dist(d,e))
    two_b=(calculate_euclidian_dist(d,f))
    two_c=(calculate_euclidian_dist(d,g))
    two_d=(calculate_euclidian_dist(e,f))
    two_e=(calculate_euclidian_dist(e,g))
    two_f=(calculate_euclidian_dist(f,g))
    two_list.append(two_a)
    two_list.append(two_b)
    two_list.append(two_c)
    two_list.append(two_d)
    two_list.append(two_e)
    two_list.append(two_f)
    p=sorted(two_list, key=lambda x: x.imag) 
    print("Minimum distance of cluster 2 is:",p[0])

def cluster_three():
    z=[]
    h=[9,3]
    i=[12,3]
    j=[11,4]
    k=[12,6]
    l=[10,5]
    a=(calculate_euclidian_dist(h,i))
    b=(calculate_euclidian_dist(h,j))
    c=(calculate_euclidian_dist(h,k))
    d=(calculate_euclidian_dist(h,l))
    e=(calculate_euclidian_dist(i,j))
    f=(calculate_euclidian_dist(i,k))
    g=(calculate_euclidian_dist(i,l))
    h=(calculate_euclidian_dist(j,k))
    i=(calculate_euclidian_dist(j,l))
    j=(calculate_euclidian_dist(k,l))
    z.append(a)
    z.append(b)
    z.append(c)
    z.append(d)
    z.append(e)
    z.append(f)
    z.append(g)
    z.append(h)
    z.append(i)
    z.append(j)
    p=sorted(z, key=lambda x: x.imag) 
    print("Minimum distance of cluster 3 is:",p[0])
    
if __name__=='__main__':
    # calling all function to calculate euclidian distance of each cluster.
    try:
        cluster_one()
        cluster_two()
        cluster_three()
    # applying except block to catch the execption
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")
    


# In[55]:


# solution of 2
# calculating the average of the distance between pairs of points, one from each of the two clusters.
from math import sqrt
# defining the function to calculate average distance between multiple points
def avg_distance(x,y):
    n = len(x)
    dist = 0
    for i in range(n):
        xi = x[i]
        yi = y[i]
        for j in range(i+1,n):
            dx = x[j]-xi
            dy = y[j]-yi
            dist += sqrt(dx*dx+dy*dy)
    return 2.0*dist/(n*(n-1))

# defining function to calculate average distance of first cluster
def avg_dist_first_cluster():
    x=[2,5,3]
    y=[2,2,4]
    return avg_distance(x,y)

# defining function to calculate avarage distance of second cluster
def avg_dist_second_cluster():
    x=[9,12,11,12,10]
    y=[3,3,4,6,5]
    return avg_distance(x,y)
# defining function to calculate average distance of third cluster
def avg_dist_third_cluster():
    x=[4,6,7,4]
    y=[8,8,10,10]
    return avg_distance(x,y)

    
# function calling and storing value on variables.    
c1=avg_dist_first_cluster()
c2=avg_dist_second_cluster()
c3=avg_dist_third_cluster()

# defining function to calculate average distance of two clusters.
def average(number1, number2):
    avg=(number1 + number2) / 2.0
    return avg
# claculating average distance between two clusters
if __name__=='__main__':
    try:
        print("Average distance between cluster C1 and Cluster C2",average(c1,c2))
        print("Average distance between cluster C1 and Cluster C3",average(c1,c3))
        print("Average distance between cluster C2 and Cluster C3",average(c2,c3))
    # applying except block to catch the execption
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")
        
        

