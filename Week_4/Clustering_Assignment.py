#!/usr/bin/env python
# coding: utf-8

# In[40]:



# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Assignment No:3-Clustering


# In[39]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import traceback
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import numpy as np
from numpy.lib import scimath


# defining function to load iris data 
def loading_iris():

    iris_data = load_iris()
    df_elbow = pd.DataFrame(iris_data.data, columns=iris_data['feature_names'])

    df_data= pd.DataFrame(iris_data.data)
    # taking only four values of datasets
    df_gap_data = df_data[[0, 1, 2, 3]].values

    return df_elbow, df_gap_data


# defining function to plot elbow
def plot_elbow(x):
  
    plt.figure()
    plt.plot(list(x.keys()), list(x.values()))
    plt.xlabel('Cluster')
    plt.ylabel('SSD')
    plt.title('SSD VS Elbow plot')
    plt.show()

    return


# defining function to evaluate elbow for k
def evaluate_elbow(iris_data, k):
   
    ssd = {}
    for i in range(1, k + 1):
        kmeans = KMeans(n_clusters=i, max_iter=2000).fit(iris_data)
        iris_data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        ssd[i] = kmeans.inertia_

    # calling plot function
    plot_elbow(ssd)

    return



# defining function to calculate gap statistics for ks range
def get_gap_statistics(data, refs=None, nrefs=30, ks=range(1, 11)):
    # calculating distance
    dst = scipy.spatial.distance.euclidean
    # defining shape
    shape = data.shape
    # checking condition for given refs
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(np.diag(tops-bots))
        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i]*dists+bots
    else:
        rands = refs

    gaps = np.zeros((len(ks),))
    errors = np.zeros((len(ks),))
    labels = dict((el,[]) for el in ks)
    for (i, k) in enumerate(ks):
        (kmc, kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])
        labels[k] = kml

        refdisps = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc, kml) = scipy.cluster.vq.kmeans2(rands[:, :, j], k)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])

        # Computing  gaps
        gaps[i] = scimath.log(np.mean(refdisps))-scimath.log(disp)

        # Computing errors
        errors[i] = scimath.sqrt(sum(((scimath.log(refdisp)-np.mean(scimath.log(refdisps)))**2)
                                     for refdisp in refdisps)/float(nrefs)) * scimath.sqrt(1+1/nrefs)

    xval = range(1, len(gaps) + 1)
    yval = gaps
    plt.errorbar(xval, yval, xerr=None, yerr=errors)
    plt.xlabel('K Clusters')
    plt.ylabel('Gap_Statistics')
    plt.title('Gap Statistics for : nref={}'.format(nrefs))
    plt.show()

    return


if __name__ == '__main__':

    try:
        # Load iris data
        elbow,gap  = loading_iris()

        # calling elbow method
        evaluate_elbow(elbow, k=10)

        # Gap Statistics for 20 number of reference distribution
        get_gap_statistics(gap, refs=None, nrefs=20, ks=range(1, 11))

        # Gap Statistics for 30 number of reference distribution
        get_gap_statistics(gap, refs=None, nrefs=30, ks=range(1, 11))
        
        # Gap Statistics for 40 number of reference distribution
        get_gap_statistics(gap, refs=None, nrefs=40, ks=range(1, 11))
        
        # Gap Statistics for 50 number of reference distribution
        get_gap_statistics(gap, refs=None, nrefs=50, ks=range(1, 11))

    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('\nAn exception of type {0} occurred. Arguments:\n{1!r}'.format(type(exception).__name__, exception.args))
    finally:
        print()
        
    print("Solution 1")
    print('According to the plot, optimum value of k is 3')
    print('According to the gap statistics,optimum value of k is also 3')

    print("Solution 2")
    print("Values of both methods are same")

   
    print('Solution 3')
    print('The gap statistic compares the total within intra-cluster variation for different values of k with their expected'
          'values under null reference distribution of the data. Hence gap statistics is \nbetter than elbow method ')

