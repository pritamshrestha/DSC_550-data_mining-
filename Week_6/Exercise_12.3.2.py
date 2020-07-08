#!/usr/bin/env python
# coding: utf-8

# Q:The following training set obeys the rule that the positive examples all have vectors whose components sum to 10 or more,
#     while the is less than 10 for the negative examples:
#         ([3,4,5],+1) ([2,7,2],+1) ([5,5,5],+1)
#          ([1,2,3],-1) ([3,3,2],-1) ([2,4,1],-1)
# a) Which of these six vectors are the support vectors?

# In[ ]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/08/2020
# Exercise No:12.3.2
# Reference:Mining of massive dataset


# In[14]:


import traceback
# defining function to calculate shortest distance from the plane to that point
def get_dist(a,b,c):
    # applying formula of distance using given quation x+y+z=d
    dist=(a+b+c-10)/3**0.5
    return dist
# calling from the main function

if __name__=='__main__':
    # calling all function to calculate euclidian distance of each cluster.
    try:
        print(get_dist(3,4,5))
        print(get_dist(2,7,2))
        print(get_dist(5,5,5))
        print(get_dist(1,2,3))
        print(get_dist(3,3,2))
        print(get_dist(2,4,1))
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")

    


# In[ ]:


# From the above result, there are six shortest destance from the hyperplane and three from positive side and 3 from 
# negative side. Now, shortest point from positive side is (2,7,2) and from negative side is (3,3,2).
# Hence, point [2,7,2] and [3,3,2] are support vectors.

