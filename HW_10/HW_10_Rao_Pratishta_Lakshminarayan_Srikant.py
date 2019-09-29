"""
Title: HW_10_Rao_Pratishta_Lakshminarayan_Srikant.py
Course: CSCI 720
Date: 04/15/2019
Author: Pratishta Prakash Rao, Srikant Lakshminarayan
Description: Code to implement the DBscan algorithm
"""

import pandas
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys


def plot_data(data):
    """
    Function to plot the given data
    :param data: data points
    """
    # Getting the values for each column
    col_1 = data.iloc[:, 1]
    col_2 = data.iloc[:, 2]
    col_3 = data.iloc[:, 3]

    # Creating an array of lists with each list containing
    # a single row values for all the columns in the data frame
    data_points = np.array(list(zip(col_1, col_2, col_3)))

    # 3D plot for the given data
    cm = plt.get_cmap("RdYlGn")
    c = np.abs(col_3)
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    p3d = ax3D.scatter(col_1, col_2, col_3, s=30, c=c, marker='o')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('3D plot for the given data')
    plt.show()

    # 2D plot for the given data
    marker_size = 15
    plt.scatter(data_points[:, 0], data_points[:, 1], marker_size, c=data_points[:, 2])
    plt.title('2D plot for the given data')
    plt.show()


def plot_figure(data, distance_list):
    """
    Function to plot figure of point vs distance
    :param data:            data points
    :param distance_list:   list of the distances found between data points
    """
    plt.figure()
    list_colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'magenta', 'black', 'orange', 'pink', 'purple',
                   'gray']
    x_axis=[number for number in range(len(data))]
    for nth_neighbor in range(10):
        nth_neighbor_list=[]
        for each_dist_list in distance_list:
            nth_neighbor_list.append(each_dist_list[nth_neighbor][1])
        print("plotting")
        print(nth_neighbor_list)
        nth_neighbor_list.sort()
        print(nth_neighbor_list)
        plt.plot(x_axis, nth_neighbor_list, list_colors[nth_neighbor], linestyle='-', linewidth=1)

    plt.xlabel('Point')
    plt.ylabel('Distance')
    plt.title('Plot for Point vs Distance ')
    plt.show()


def db_scan(distance_list, data):
    """
    Function to implement dbscan algorithm.
    :param distance_list: list of the distances found between data points
    """

    epsilon=1.3
    min_points=8

    # variable to store cluster id
    cluster_number=0

    # variable to store if the points are noise or core points
    # 0-undefined
    # -1-noise
    # any other number will be the cluster id
    labels=[0 for temp in range(len(distance_list))]
    for data_point_index in range(len(distance_list)):
        # print(data_point_index,labels[data_point_index])
        if(labels[data_point_index]!=0):
            continue
        neighbors=get_neighbors(distance_list, data_point_index, epsilon)
        if(min_points>len(neighbors)):
            labels[data_point_index]=-1
            continue

        cluster_number+=1
        labels[data_point_index]=cluster_number

        for each_point in neighbors:
            if(labels[each_point[0]]==-1):
                labels[each_point[0]]=cluster_number
            if(labels[each_point[0]]!=0):
                continue


            labels[each_point[0]] = cluster_number
            new_neighbors=get_neighbors(distance_list, each_point[0], epsilon)

            if(len(new_neighbors)>=min_points):
                for neighs in new_neighbors:
                    if(neighs not in neighbors):
                        neighbors.append(neighs)

    for l in labels:
        print(l)
    # print("max",max(labels))
    plot_clusters(labels, data)


def get_neighbors(distance_list, data_point_index, epsilon):
    """
    Function to get the neighbors of a particular data point within specified distance.
    :param distance_list:       list of the distances found between data points
    :param data_point_index:    index of the current point
    :param epsilon:             radius for the cluster
    :return:                    neighboring data points which are epsilon distance from the
                                core point
    """
    neighbor_list = distance_list[data_point_index]
    new_neighbor_list = []
    for neighbors in neighbor_list:
        if (neighbors[1] < epsilon):
            new_neighbor_list.append(neighbors)
    # print("new_neighbor_list",new_neighbor_list)
    return new_neighbor_list


def center_mass(list_points_clusters, data):
    """
     Function to calculate the center of mass
    :param list_points_clusters:    all clusters as list
    :param data:                    dataframe of the whole data
    :return:                        list of center of mass of clusters
    """
    list_points_clusters.sort(key=len)
    centerMassList=[]
    for cluster in list_points_clusters:
        print("Number of data points in the cluster:" + str(len(cluster)))
        cluster_points=[]
        for indexes in cluster:
            cluster_points.append(data[indexes][1:])
        centerMassList.append(np.asarray(cluster_points).mean())
    return centerMassList


def plot_clusters(labels, data):
    """
    Plot different clusters after getting results of dbscan.
    :param labels:           list of labels i.e. core or noise.
    :param data:             dataframe of the whole data
    """
    unique_labels = []
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    for x in labels:
        # check if exists in unique_list or not
        if x not in unique_labels:
            unique_labels.append(x)

    # variable to store all indexes of different clusters
    list_points_clusters = []
    noise_points = []
    for label in unique_labels:
        if (label == -1):
            noise_points.extend([i for i, v in enumerate(labels) if v == label])
        if (label != -1):
            list_points_clusters.append([i for i, v in enumerate(labels) if v == label])
            new_list_points = [i for i, v in enumerate(labels) if v == label]
            x_list = []
            y_list = []
            z_list = []
            for index_point in new_list_points:
                point = data.iloc[index_point]
                x_list.append(point[1])
                y_list.append(point[2])
                z_list.append(point[3])
            cm = plt.get_cmap("RdYlGn")
            c = np.abs(z_list)

            p3d = ax3D.scatter(x_list, y_list, z_list, s=30, c=c, marker='o')
    ax3D.set_xlabel("X-axis")
    ax3D.set_ylabel("Y-axis")
    ax3D.set_zlabel("z-axis")
    plt.title("Plot for the clusters formed")
    plt.show()

    # for sub_list in list_points_clusters:
    #     cluster_points = []
    #     for indexes in sub_list:
    #         cluster_points.append(data.iloc[indexes][1:])
    #     x_axis = cluster_points[0][0]
    #     y_axis = cluster_points[0][1]
    #     z_axis = cluster_points[0][2]


def main():
    """
    Main function to read in file and find euclidean distance between each point to every other data point.
    """

    # variable to distances of each point to every other point.The list at position one will
    # have distances of 1st point in data file to rest of the points
    distance_list=[]

    # read in csv file
    # Usage: python HW**.py dbscan.csv
    data = pandas.read_csv(sys.argv[1])

    # Plot the given data points
    plot_data(data)

    # computation for finding euclidean distance between each data point to every other data point
    for index_i, row_i in data.iterrows():
        each_point_dist=[]
        for index_j, row_j in data.iterrows():
            if index_i!= index_j:

                dist= distance.euclidean(row_i[1:],row_j[1:])
                # print(dist)
                each_point_dist.append((index_j,dist))
        distance_list.append(each_point_dist)

    # print(distance_list[0])
    for index in range(len(distance_list)):
        distance_list[index].sort(key=lambda tup: tup[1])
    # print(distance_list[0])

    plot_figure(data, distance_list)
    db_scan(distance_list, data)


if __name__ == '__main__':
    main()
