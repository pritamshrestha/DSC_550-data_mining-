#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Name:Pritam Shrestha
# FileName:Exercise_3.2.1
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:Uses of shingles in the sentence
# Due Date:06/20/20
# Assignment No:2.1
# Reference:https:https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/


# In[16]:


# Questions: What are the first ten 3-shingles in the first sentence of section 3.2?


# In[8]:


# Solution:
#initializeing the variable 
testing_sentence='The most effective way to represent documents as sets, for the purpose of identifying lexically similar documents is to construct from the document the set of short strings that appear within it.'
# given value of k
k = 3

# main function to execute the block of code
def main():
    # try block for execution
    try:
        # printing given text
        print('Given sentence == ', testing_sentence, '\n')
        # splitting after replacing comma and storing data on cleaned_sentence variable
        cleaned_sentence = testing_sentence.replace(',', '').split()
        print("First ten 3-shingles in text ==> " + str([cleaned_sentence[x:x + k] for x in range(0, 10)]))  # Print 3-shingles
    except Exception as exception:
        # printing the exception in nice way.
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args));  
    finally:
        print("finally block is executed wheather exception is handled or not!!")

if __name__ == '__main__':
    # calling function from main function
    main()


# In[ ]:


#Thanks!!!

