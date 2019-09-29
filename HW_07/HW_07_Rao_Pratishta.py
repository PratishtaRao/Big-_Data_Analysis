"""
Title: HW_07_Rao_Pratishta_Trainer.py
Course: CSCI 720
Date: 03/22/2019
Author: Pratishta Prakash Rao
Description: Code to implement the K-Means clustering algorithm
"""

import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import time


def euclidean_distance(data_array, cluster, axis=1):
    """
    Function to calculate the euclidean distance using numpy
    :param data_array:  set of data points
    :param cluster:     centroids of each cluster
    :return:            the euclidean distance of the point to each cluster
    """
    distance = np.linalg.norm(data_array - cluster, axis=axis)
    return distance


def k_means(data, k):
    """
    Function to implement the k means clustering algorithm
    :param data:    data points
    :param k:       the number of clusters to be made
    :return:        sse, centroid for each cluster, number of poijnts for each cluster, execution time
    """
    k = k + 1
    centroids = list()
    # Creating centroids for k clusters
    for num in range(k):
        centroids.append(np.random.randint(0, np.max(data) , size=4))
    # Creating a numpy array for the obtained centroids
    centroids_array = np.array(centroids, dtype=np.float32)
    # Initializing variable to keep track of old centroids
    old_centroids = np.zeros(centroids_array.shape)
    # Initializing variable to find clusters
    clusters = np.zeros((data.shape[0], 1))

    cluster_data_count = [0] * k
    start_time = time.time()
    while not stopping_criteria(old_centroids, centroids_array):

        # For all data points...
        for val in range(len(data)):
            # Calculating the euclidean distance
            distances = euclidean_distance(data[val], centroids_array)
            # Finding the minimum distance
            min_distance = np.argmin(distances)
            # Assigning the cluster number to the data point
            clusters[val] = min_distance

            # print(clusters[val], data[val])

        # To keep Track of the old centroids
        old_centroids = deepcopy(centroids_array)

        # Recomputing the new values of the centroids
        # Getting the data points in each cluster
        for num in range(k):
            array_points = list()
            for value in range(len(data)):
                if (clusters[value][0] == num):
                    array_points.append(list(data[value]))
            cluster_data_count[num] = len(array_points)
            if len(array_points) != 0:
                centroids_array[num] = np.mean(array_points, axis=0)
            else:
                centroids_array[num] = np.random.randint(0, np.max(data), size=4)
    # To calculate the time taken for the runtime
    time_taken = time.time() - start_time
    sse = get_sum_of_squared_errors(clusters, centroids_array, data)
    return sse, cluster_data_count, centroids_array, time_taken


def get_sum_of_squared_errors(clusters, centroids, data):
    """
    Function to get the sum squared value for the given data points
    :param clusters: number of clusters
    :param centroids: final centroids calculated
    :param data:    data points
    :return:
    """

    sse = [0] * (int(np.amax(clusters)) + 1)
    for val in range(len(clusters)):
        cluster_idx = int(clusters[val][0])
        sse[cluster_idx] += euclidean_distance(data[val], centroids[cluster_idx], axis=0)**2

    return sse


def stopping_criteria(old_centroids, new_centroids):
    return (np.isclose(old_centroids, new_centroids)).all()


def compare_kmeans(data, k):
    """
    Calculates for all K values from 1 to 15
    :param data: data points
    :param k:  number of clusters
    :return:
    """
    sse_record = list()
    cluster_count_record = list()
    sse_individual_record = list()
    centroids_record = list()
    time_record = list()

    for cluster_num in range(k + 1):
        best_sse = sys.maxsize
        best_cluster_count = list()
        best_see_individual = list()
        best_centroids = list()
        best_time = 0.0

        for itr in range(1000):
            sse, cluster_count, centroids, time_taken = k_means(data, cluster_num)
            sse_total = sum(sse)
            if sse_total < best_sse:
                best_sse = sse_total
                best_cluster_count = cluster_count
                best_see_individual = sse
                best_centroids = centroids
                best_time = time_taken

        sse_record.append(best_sse)
        cluster_count_record.append(best_cluster_count)
        sse_individual_record.append(best_see_individual)
        centroids_record.append(best_centroids)
        time_record.append(best_time)
    plotter(sse_record, time_record, k)


def plotter(sse, time, k):
    """
    Function to plot graphs
    :param sse:  sum squared error
    :param time: runtime of the program
    :param k:    the number of clusters to be made
    :return:
    """
    k_range = list(range(k + 1))
    plt.plot(k_range, time)
    plt.title("K vs Execution time")
    plt.xlabel("K")
    plt.ylabel("Time taken (s)")
    plt.show()

    plt.plot(k_range, sse)
    plt.title("K vs SSE")
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.show()


def main():
    # Usage: python HW07.py  k_means.csv
    data = p.read_csv(sys.argv[1], header=None)
    # Getting the values for each column
    col_1 = data.iloc[:, 1]
    col_2 = data.iloc[:, 2]
    col_3 = data.iloc[:, 3]
    col_4 = data.iloc[:, 4]
    # Creating an array of lists with each list containing
    # a single row values for all the columns in the data frame
    data_points = np.array(list(zip(col_1, col_2, col_3, col_4)))
    compare_kmeans(data_points, 15)


if __name__ == '__main__':
    main()
