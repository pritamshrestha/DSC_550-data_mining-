#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Name:Pritam Shrestha
# FileName:closed_itemsets
# Date:6/05/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:implementation of itemsets mining
# Due Date:06/20/20
# Assignment No:2.1
# Reference:https:http://www.dataminingbook.info/uploads/videos/lecture4/


# In[31]:


import numpy as np
import math
import matplotlib.pyplot as plt


# itemsets mining implementing CHARM algorithm 
# insted of the minsup i have passed two minsup as 3000 and 5000 to calculate the itemsets.

""" This program creates the brute force algorithm
    for itemset mining. This algorithm is detailed
    on page 223 in Data Mining and Machine Learning
"""
import sys
import pandas as pd
filename="mushroom.txt"
def create_dict_from_file(filename):
    """ Read in a file of itemsets
        each row is considered the transaction id
        and each line contains the items associated
        with it.
        This function returns a dictionary that
        has a key set as the tid and has values
        of the list of items (strings)
    """
    f = open(filename, 'r')
    d = {}
    for tids, line_items in enumerate(f):
           d[tids] = [j for j in line_items.split(' ')
                           if j != '\n']
    return d
def create_database(itemset):
    "Uses dummy indexing to create the binary database"
    return pd.Series(itemset).str.join('|').str.get_dummies()
def compute_support(df, column):
    "Exploits the binary nature of the database"
    return df[column].sum()

if __name__ == '__main__':

    dict_itemset = create_dict_from_file(filename)
    database = create_database(dict_itemset)
    # Executes the brute force algorithm
    # NOTE: a list comprehension is faster
    # Check if the command line arguments are given
    try:
        # function calling inside the try block to catch the error
        output(3000)
        output(5000)
    except:
        print('You need both a filename and a minimum support value!')
    dict_itemset = create_dict_from_file(filename)
    database = create_database(dict_itemset)
    # Executes the brute force algorithm
    # NOTE: a list comprehension is faster
def output(minsup):
    print("Filename",filename)
    print("Minsup",minsup)
    freq_items = []
    for col in database.columns:
        sup = compute_support(database, col)
        if sup >= minsup:
            freq_items.append(int(col))
            print(freq_items,sup)
        else:
            pass
    print('There are %d items with frequency'          ' greater than or equal to minsup.' % len(freq_items))
    #print((freq_items),sup[i])


# In[ ]:


# Thanks!!!

