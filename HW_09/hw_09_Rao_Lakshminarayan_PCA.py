"""
Title: HW_08_Rao_Pratishta.py
Course: CSCI 720
Date: 03/31/2019
Author: Pratishta Prakash Rao, Srikanth Lakshminarayan
Description: Code to perform the Principal component analysis on the given data
"""


import numpy as np
import pandas
import matplotlib.pyplot as plt
from heapq import nlargest
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import sys
from scipy.cluster.vq import vq


def pca_analysis(val):
    """
    Function to find the eigen values and vectors

    :param val: the data points of the given csv file
    :return:    eigen values, eigen vectors
    """
    cov_val = np.cov(val, rowvar=False)
    eig_values, eig_vectors = np.linalg.eig(cov_val)
    return eig_values, eig_vectors


def normalize_eigen(eig_values):
    """
    Function to normalize the eigen values

    :param eig_values: eigen values obtained for the data set
    """

    abs_eigen_value = np.absolute(eig_values)
    sort_eigen = sorted(abs_eigen_value)
    sum_eigen = np.sum(sort_eigen)
    # Divide by sorted eigen values
    norm_eigen = np.divide(sort_eigen, sum_eigen)
    plot_eigen(norm_eigen)


def plot_eigen(eigen_values):
    """
    Function to plot the eigen values

    :param eigen_values: eigen values obtained for the data set
    """
    eigen_values = np.cumsum(eigen_values)
    x_label = list(range(np.shape(eigen_values)[0]))
    y_label = eigen_values.tolist()
    plt.plot(x_label, y_label)
    plt.ylabel('Normalized eigen values')
    plt.xlabel('Count')
    plt.title('Plot of Cumulative Sum of noramlized eigen values')
    plt.show()


def analysis_eigen(eig_val, eig_vectors):
    """
    Function to add the obatined eigen values and eigen vectors
    to a dictonary and get eigen vectors of the four maximum
    eigen values

    :param eig_val:  eigen values obatined for the data set
    :param eig_vectors: eigen vectors obtained for the data set
    :return:  the eigen vectors of the four maximum eigen values
    """
    eig_val = eig_val.tolist()
    eig_vectors = np.round(eig_vectors, decimals=1)
    eig_vectors = eig_vectors.tolist()
    eigen_dict = {}
    for val in range(len(eig_val)):
        eigen_dict[eig_val[val]] = eig_vectors[val]
    eigen_dict = dict(zip(eig_val, eig_vectors))
    return get_max_values(eigen_dict)


def get_max_values(dict_values):
    """
    Function to get the eigen vectors for four maximum eigen values
    :param dict_values: dictonary containing eigen values and eigen vectors
    :return:   the eigen vectors of the four maximum eigen values
    """
    max_values = nlargest(4, dict_values)
    print("The first four eigen vectors associated with the four highest eigen values:")
    mask = list()
    for key in max_values:
        mask.append(dict_values[key])
    max_eigen_vectors = np.asarray(mask)
    print(max_eigen_vectors)
    return max_eigen_vectors


def data_project(values, max_eigen):
    """
    Function to obtain the dot product of the data points with the
    eigen vectors.

    :param values:   the data points of the given csv file
    :param max_eigen: the eigen vectors of the four maximum eigen values
    :return:  a numpy array with dot product of the data points with the eigen vectors.
    """
    trans_values = np.transpose(values)
    resultant = np.matmul( max_eigen, trans_values)
    # Delete the last row in the array
    resultant = np.delete(resultant, 3, 0)
    plot_3d(resultant)
    return resultant


def plot_3d(matrix):
    """
    Function to plot the 3D data

    :param matrix: a numpy array with dot product of the data points with the eigen vectors.
    """
    x = matrix[0, :].tolist()
    y = matrix[1, :].tolist()
    z = matrix[2, :].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def k_means(matrix):
    """
    Function to perform the k means clustering and plot the
    k vs sse obtained
    :param matrix:  a numpy array with dot product of the data points with the eigen vectors.
    """
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(np.transpose(matrix))
        sse[k] = kmeans.inertia_
        # Inflection point found at k = 6, so
        # Cluster counts for when k = 6
        if k == 6:
            labels = kmeans.labels_
            cluster_size = np.bincount(labels)
            centroids = kmeans.cluster_centers_
            print("Average prototype for k = 6")
            print(centroids)
            print("Cluster Size:")
            print(cluster_size)
            print("Sorted cluster size:")
            print(np.sort(cluster_size))


    # plt.figure()
    # plt.plot(list(sse.keys()), list(sse.values()))
    # plt.xlabel("Number of cluster")
    # plt.ylabel("SSE")
    # plt.title("K means vs SSE")
    # plt.show()


def main():
    # Usage: python HW07.py  PCA.csv
    # To read the city and country into a data frame
    data = pandas.read_csv(sys.argv[1])
    # Convert the data frame into a numpy array
    data = data.drop(['ID'], axis=1)
    values = data.values
    # Get Eigen values and eigen vectors
    result_eig_val, result_eigen_vector = pca_analysis(values)
    # Normalise eigen values
    normalize_eigen(result_eig_val)
    # Analyze the eigen vectors and values
    result_vectors = analysis_eigen(result_eig_val, result_eigen_vector)
    # Data projection
    product = data_project(values, result_vectors)
    # K means clustering on the new data
    k_means(product)


if __name__ == '__main__':
    main()


