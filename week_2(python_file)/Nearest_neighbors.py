#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Name:Pritam Shrestha
# FileName:Nearest_Neighbors
# Date:6/19/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:finding nearest neighbors and ration of nearest and farthest neighbors
# Due Date:06/20/20
# Assignment No:2.1


# In[33]:


# Import important libraries
import random
import matplotlib.pyplot as plt


#Function to create 100 x 10000 random uniform matrix
def find_neighbors(d):
    
    hypercube= []
    for x in range(len(d)):
        uniform = []
        for n in range(10000):
            uniform.append(random.uniform(0, 1))
        hypercube.append(uniform)

    return hypercube

#Function to get closest number from 0.5 in the list
def closest(list, k=0.5):
    
    return list[min(range(len(list)), key=lambda i: abs(list[i]-k))]


#Function to get farthest number from 0.5 in the list
def farthest(list, k=0.5):
    return list[max(range(len(list)), key=lambda i: abs(list[i]-k))]

# function to get list of distance from closest and farthest point for each value of d
def get_nearest_neighbors(hypercube):
    near = []
    far = []
    for i in range(len(hypercube)):
        
        # Appending with absolute difference value from closest point
         near.append(abs(0.5-closest(hypercube[i]))) 
            
        # Append with absolute difference value from closest point
         far.append(abs(0.5-farthest(hypercube[i])))   
    return near, far

# creating function for plot of nearest and farthest 
def plot(d, near, far):
    # Plot closest distances for each d
    plt.plot(d, near)
    plt.xlabel('Dimension - d')
    plt.ylabel('Dist of nearest dn')
    plt.title('Distance from Closest point VS d')
    plt.show()

    # Plot farthest distances for each d
    plt.plot(d, far)
    plt.xlabel('Dimension - d')
    plt.ylabel('Dist of farthest df')
    plt.title('Distance from Farthest point VS d')
    plt.show()

    return


if __name__ == '__main__':
    try:
    
# dimensions for range(1,100)
        d = [x for x in range(1, 100)]     
        hypercube = find_neighbors(d)           
        near, far = get_nearest_neighbors(hypercube)          
        plot(d, near, far)  
        
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally block is executed whether exception is handled or not!!")
    

