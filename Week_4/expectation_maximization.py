#!/usr/bin/env python
# coding: utf-8

# Write a script that implements the Expectation-Maximization (EM) algorithm for clustering (see Algorithm 13.3 in Chapter 13). Run the code on the iris.txt dataset. Use the first four attributes for clustering, and use the labels only for the purity-based clustering evaluation (see below). In your implementation, you should estimate the full covariance matrix for each cluster.
# 
# For EM initialization, use the first n/k points for cluster 1, the next n/k for cluster 2, and so on. For convergence testing, you can compare the sum of the euclidean distance between the old means and the new means over the k clusters. If this distance is less than ϵ=0.001 you can stop the method.
# 
# Your program output should consist of the following information:
# 
# The final mean for each cluster
# The final covariance matrix for each cluster
# Number of iterations the EM algorithm took to converge.
# Final cluster assignment of all the points, where each point will be assigned to the cluster that yields the highest probability P(Ci|xj)
# Final size of each cluster
# Finally, you must compute the 'purity score' for your clustering, computed as follows: Assume that Ci denotes the set of points assigned to cluster i by the EM algorithm, and let Ti denote the true assignments of the points based on the last attribute. Purity score is defined as:
# 
# Purity=1n∑i=1kmaxkj=1{Ci∩Tj}

# In[41]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Assignment No:1( EM Algorithms)
# Reference:https://github.com/tamaksHarsh/emClustering and two of peer
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
from numpy.core.umath_tests import matrix_multiply as mm
import traceback

# used references of two peers to complete this exercise.
def read_data(filename):
    return pd.read_csv(filename, header=None)


def Random_val(x):
    min_value = x.min()
    max_value = x.max()

    # Generate a random number from a uniform distribution of the min and max of the column.
    return np.random.uniform(min_value, max_value, 1)[0]

# defining function to implement EM_algorithm using gaussian mixture model
def Apply_EM(iris, k, e):
    # lenth of data
    n = len(iris)

    # E-Step

    # Initializing value up to k
    sigma = np.array([np.eye(4)] * k)

    cluster_mu = []
    cluster_p = []
    for i in range(k):
        atr_mu = []
        for column in iris[[0, 1, 2, 3]]:
            atr_mu.append(Random_val(iris[column]))
        cluster_mu.append(atr_mu)
        cluster_p.append(1/k)

    cluster_mus = np.array(cluster_mu)

    like_old = 0
    i = 0
    diff = 1
# applying the given condition
    while diff > e and i < 1000000:
        ws = np.zeros((k, n))

        # calculating probability of each cluster
        for j in range(k):
            ws[j, :] = cluster_p[j] * mvn(cluster_mus[j], sigma[j]).pdf(iris.loc[:, 0:3])
        ws /= ws.sum(0)

        # M Step

        # update probabilities
        cluster_p = ws.sum(axis=1)
        cluster_p /= n

        cluster_mus = np.dot(ws, iris.loc[:, 0:3])
        cluster_mus /= ws.sum(1)[:, None]

        # update sigmas
        sigma = np.zeros((k, 4, 4))

        for j in range(k):
            # get values from data frame, subtract mean values and convert to numpy array
            ys = (iris.loc[:, 0:3] - cluster_mus[j, :]).to_numpy()

            # Calculate sigmas
            sigma[j] = (ws[j, :, None, None] * mm(ys[:, :, None], ys[:, None, :])).sum(axis=0)
        sigma /= ws.sum(axis=1)[:, None, None]

        # init temporary log likelihood variable
        like_new = 0

        # calculate probability for each
        for p, mu, sig in zip(cluster_p, cluster_mus, sigma):
            like_new += p * mvn(mu, sig).pdf(iris.loc[:, 0:3].to_numpy())

        like_new = np.log(like_new).sum()

        diff = np.abs(like_new - like_old)
        like_old = like_new

        # incrementing by 1
        i += 1

    print("\nNumber of iterations for the convergence is = ", i)
    new_nodes = pd.DataFrame()
    for node, point in enumerate(ws):
        new_nodes[node] = point

    new_nodes['tag'] = new_nodes.idxmax(axis=1)

    print("Node of clusters=", new_nodes.groupby(['tag']).agg('count')[0])

    print("Mean of mattrix for 3 cluster=\n", cluster_mus)

    print("Covariance=\n", sigma)

# calculating purity

    # Add flower(label) data  type in the dataset
    new_nodes['Type'] = iris.iloc[:, 4]

    # Grouping to get max count
    groupd = new_nodes.groupby(['Type', 'tag']).agg({'tag': ['count']})
    groupd.columns = ['tag_count']
    groupd = groupd.groupby(['Type']).agg({'tag_count': ['max']})
    groupd.columns = ['tag_count_max']
    groupd = groupd.reset_index()

    print('Purity of clustering=', round(sum(groupd['tag_count_max'])/len(iris), 2))

    return


if __name__ == '__main__':

    # Given parameters
    filename = 'iris.txt'
    clusters = 3
    epsilon = 0.001

    # Read the text file
    data = read_data(filename)

    try:
        # calling em algorithm function
        Apply_EM(data, clusters, epsilon)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred. Arguments:\n{1!r}'.format(type(exception).__name__, exception.args))
    finally:
        print("finally block is executed whether exception is handled or not!!")

