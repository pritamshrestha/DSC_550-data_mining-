#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name:Pritam Shrestha
# FileName:Non_derivable_itemsets
# Date:06/17/2020
# Course: DSC550-Data Mining
# Professor/Instructor:Brant Abeln
# Description:Non Derivable Itemsest
# Due Date:06/21/20
# Assignment No:2


# In[15]:


# Import important libraries
import pandas as pd
from itertools import chain, combinations

# creating constructer
class Main_data:
    def __init__(self):
        self.item_set = []
        self.list_set = []
        # self.st = set()
        self.item_set_df = pd.DataFrame(columns=['set_list', 'support_list'])
        self.list_set_df = pd.DataFrame(columns=['set_list'])
# reading file and converting into dataframe
    def read_first_text(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n').split(' - ')
                self.item_set.append({'set_list': line[0], 'support_list': line[1]})
        item_set_df = pd.DataFrame(self.item_set)
        item_set_df['set_list'] = item_set_df['set_list'].apply(lambda p: set(p))
        return item_set_df

    def read_second_text(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                self.st = set()
                line = line.rstrip('\n').split(' ')
                for element in line:
                    self.st.add(int(element))
                self.list_set.append({'set_list': self.st})
        list_set_df = pd.DataFrame(self.list_set)
        return list_set_df


class Check_Non_Derivable:

    @staticmethod
    def powerset(s):
        l = list(s)
        return chain.from_iterable(combinations(l, r) for r in range(len(l) + 1))

    def get_support(self, file1, sst):
        for row in file1.itertuples():
            rw = row[0]
            #if sst == row[0]:
            #    return row[1]
            #else:
            #    continue
               #    return sst # row['support']

        return type(row[0])

    def cal_bound(self, main_set, subs, file1):
        """
        :param main_set: Primary set from ndi.txt
        :param subs:  Subset of primary set
        :param pwr_set: List of all subsets of primary set
        :param file1: itemsets txt file for support
        :return:
        """
        bound = 0
        supp_st = 0
        pwr_set1 = self.powerset(main_set)
        for st in pwr_set1:
            sst = set(st)
            #print(sst)
            if subs.issubset(sst) and len(sst) < len(main_set):
                coef = pow(-1, ((len(main_set) - len(st)) + 1))
                get = self.get_support(file1, sst)
                supp_st += 10
                #supp_st += coef * 10
                # print(coef, subs)
                print(get)
        return supp_st

    def evaluate_support_and_bound(self, file1, file2):
        for index, row in file2.iterrows():
            main_set = row['set_list']
            pwr_set = self.powerset(main_set)
            for sub in pwr_set:
                subs = set(sub)
                len_diff = len(main_set) - len(subs)
                #sup_st = file1['support'][if (file1['sets'] < subs): False;].values
                #print(sup_st)
                # print(len(main_set), len(subs), len(main_set) - len(subs), subs)
                if len(subs) == 0:
                    continue
                if len_diff == 0:
                    continue
                    #checking even number
                elif (len_diff % 2) == 0: 
                    # for Lower Bound
                    lbound = self.cal_bound(main_set, subs, file1)
                    print('Lower_bound', lbound)
                else: 
                    # for odd number
                    # for Upper Bound
                    ubound = self.cal_bound(main_set, subs, file1)
                    print('Upper_bound', ubound)
        return

if __name__ == '__main__':

    try:
        file1 = 'itemsets.txt'
        file2 = 'ndi.txt'

        data = Main_data()
        item_set_df = data.read_first_text(file1)
        list_set_df = data.read_second_text(file2)
        print(list_set_df)
        print(item_set_df.head())
    

       #calling class and function
        Check_Non_Derivable()
        Check_Non_Derivable().evaluate_support_and_bound(item_set_df, list_set_df)
        
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred.  Arguments:\n{1!r}'.format(type(exception).__name__, exception.args)); 
    finally:
        print("Finally, the block is executed whether an exception is handled or not!!")

