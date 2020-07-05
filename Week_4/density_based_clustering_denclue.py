#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Pritam shrestha
# DSC550 - Data Mining
# Date: 07/3/2020
# Assignment No:2-Density_based_clustering
# Reference: https://github.com/mgarrett57/DENCLUE/blob/master/denclue.py   


# In[17]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import traceback


def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):
    """
    This function climbs the 'hill' of the kernel density function
    and finds the 'peak', which represents the density attractor
    """
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)

    # Sum of the last three steps is used to establish radius
    # of neighborhood around attractor. Authors suggested two
    # steps works well, but I found three is more robust to
    # noisey datasets.
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1 - x_l0)
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters > 3 and error < eps:
            break
    return [x_l1, prob, radius]


def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]
    d = X.shape[1]
    superweight = 0.  # superweight is the kernel X weight for each item
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        kernel = kernel * W[j] / (h ** d)
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * X[j])
    x_l1 = x_l1 / superweight
    density = superweight / np.sum(W)
    return [x_l1, density]


def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel


class DENCLUE(BaseEstimator, ClusterMixin):
    """Perform DENCLUE clustering from vector array.
    Parameters
    ----------
    h : float, optional
        The smoothing parameter for the gaussian kernel. This is a hyper-
        parameter, and the optimal value depends on data. Default is the
        np.std(X)/5.
    eps : float, optional
        Convergence threshold parameter for density attractors
    min_density : float, optional
        The minimum kernel density required for a cluster attractor to be
        considered a cluster and not noise.  Cluster info will stil be kept
        but the label for the corresponding instances will be -1 for noise.
        Since what consitutes a high enough kernel density depends on the
        nature of the data, it's often best to fit the model first and
        explore the results before deciding on the min_density, which can be
        set later with the 'set_minimum_density' method.
        Default is 0.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. In this version, I've only tested 'euclidean' at this
        moment.
    Attributes
    -------
    clust_info_: dictionary [n_clusters]
        Contains relevant information of all clusters (i.e. density attractors)
        Information is retained even if the attractor is lower than the
        minimum density required to be labelled a cluster.
    labels_ : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.
    Notes
    -----
    References
    ----------
    Hinneburg A., Gabriel HH. "DENCLUE 2.0: Fast Clustering Based on Kernel
    Density Estimation". In: R. Berthold M., Shawe-Taylor J., Lavrač N. (eds)
    Advances in Intelligent Data Analysis VII. IDA 2007
    """

    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric
        print('\n Attributes used for clustering -')
        print(' h = ', h)
        print(' minimum density = ', min_density)
        print(' epsilon = ', eps)

    def fit(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # create default values
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight

        # initialize all labels to noise
        labels = -np.ones(X.shape[0])

        # climb each hill
        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)

        # initialize cluster graph to finalize clusters. Networkx graph is
        # used to verify clusters, which are connected components of the
        # graph. Edges are defined as density attractors being in the same
        # neighborhood as defined by our radii for each attractor.
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}

        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1)
            g_clusters.nodes[j1]['attractor'] = density_attractors[j1]
            g_clusters.nodes[j1]['radius'] = radii[j1]
            g_clusters.nodes[j1]['density'] = density[j1]

        # populate cluster graph
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2):
                    continue
                diff = np.linalg.norm(g_clusters.nodes[j1]['attractor'] -
                                      g_clusters.nodes[j2]['attractor'])
                if diff <= (g_clusters.nodes[j1]['radius'] + g_clusters.nodes[j1]['radius']):
                    g_clusters.add_edge(j1, j2)

        num_clusters = 0

        # loop through all connected components
        for c in nx.connected_components(g_clusters):
            clust = g_clusters.subgraph(c)

            # get maximum density of attractors and location
            max_instance = max(clust, key=lambda x: clust.nodes[x]['density'])
            max_density = clust.nodes[max_instance]['density']
            max_centroid = clust.nodes[max_instance]['attractor']

            # populate cluster_info dict
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': len(clust.nodes()),
                                          'attractor': np.around(max_centroid.astype(np.double), 1),
                                          'density': np.around(max_density.astype(np.double), 3)}

            # if the cluster density is not higher than the minimum,
            # instances are kept classified as noise
            if max_density >= self.min_density:
                labels[clust.nodes()] = num_clusters
            num_clusters += 1

        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self.clust_info_


def calculate_purity(iris_df, clusters):
    major_class = 0
    t_size = 0
    for clust in clusters:
        n_ver = 0
        n_set = 0
        n_vir = 0
        for index, row in iris_df.iterrows():
            if index in clusters[clust]['instances']:
                if row[4] == 'Iris-versicolor':
                    n_ver += 1
                elif row[4] == 'Iris-setosa':
                    n_set += 1
                elif row[4] == 'Iris-virginica':
                    n_vir += 1

        major_class += max(n_ver, n_set, n_vir)
        t_size += clusters[clust]['size']

    print('Purity of clustering is: ', round(major_class / t_size, 2))
    return

if __name__ == '__main__':

    # Reading data 
    df = pd.read_csv('iris.txt', header=None)
    print(iris_df.head())

    # Convert the data into array
    data1 = np.array(df)
    # ampping iris data
    iris = np.mat(data1[:, 0:4])

    try:
        # Applying denclue algorithm
        denclue = DENCLUE(h=0.37, eps=0.0001, min_density=0.13, metric='euclidean')
        clusters = denclue.fit(iris, y=None, sample_weight=None)

        # Print attributes of the clusters
        print('Final Clusters are:')
        cluster = pd.DataFrame.from_dict(clusters, orient='index')
        print(cluster[['size', 'density', 'attractor']])

        # Calculate purity of cluster
        calculate_purity(df, clusters)
    except Exception as exception:
        print('exception')
        traceback.print_exc()
        print('An exception of type {0} occurred. Arguments:\n{1!r}'.format(type(exception).__name__, exception.args))
    finally:
        print("Finally block is executed whether exception is handled or not!!")


