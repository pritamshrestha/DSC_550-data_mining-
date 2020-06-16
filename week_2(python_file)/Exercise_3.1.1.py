#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Name:Pritam Shrestha
# FileName:Exercise_3.1.1
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:Uses of Jaccard similarities 
# Due Date:06/20/20
# Assignment No:2.1
# Reference:https://stackoverflow.com/questions/46975929/how-can-i-calculate-the-jaccard-similarity-of-two-lists-containing-strings-in-py


# In[38]:


# Q uestions:Compute the Jaccard similarities of each pair of the following three sets: {1,2,3,4},{2,3,5,7}and{2,4,6}.
# mathematical formaula of jaccard similarities

# if there is two sets A,B then we can use this formula to find the jaccard similarities
# j(A,B)=|A Intersection B|/|A|+|B|-|A Intersection B|


# In[40]:


# Solution:
# initializing the variables

A={1,2,3,4}
B={2,3,5,7}
C={2,4,6}

# Creating the function to calculate the jaccard similarities.

def jaccard_similarity(set1, set2):
    s1 = set(set1)
    s2 = set(set2)
    return len(s1.intersection(s2)) / len(s1.union(s2))
# defining main function to called jaccard similarities
def main():
    # calling function inside try block
    try:
        pair1=jaccard_similarity(B,C)
        pair2=jaccard_similarity(B,A)
        pair3=jaccard_similarity(C,A)
        print("The jaccard similarity between B and C:",pair1)
        print("The jaccard similarity between B and A:",pair2)
        print("The jaccard similarity between C and A:",pair3)
    # values are converted into percentage to make more readable.
        pair1_in_percentage=pair1*100
        pair2_in_percentage=pair2*100
        pair3_in_percentage=pair3*100
        print("The percentage of jaccard similarity between B and C:",pair1_in_percentage)
        print("The percentage of jaccard similarity between B and A:",pair2_in_percentage)
        print("The percentage of jaccard similarity between C and A:",pair3_in_percentage)
        
    # applying except block to catch the execption
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("finally block is executed wheather exception is handled or not!!")
        
if __name__ == '__main__':
    main()

