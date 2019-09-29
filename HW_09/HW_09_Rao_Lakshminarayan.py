"""
Program to perform agglomeration clustering on data and storing the clusters in output.txt file
"""

import pandas
import numpy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
import scipy.cluster.hierarchy as sci

def read_data():
    '''
    Function to read in data and return the each datapoint as a cluster and the dataframe.
    :return:
    '''
    data = pandas.read_csv('/Users/srinivaslakshminarayan/PycharmProjects/bda/HW_PCA_SHOPPING_CART_v850_KEY.csv')
    column_values=data.columns
    len_columns=len(column_values)

    # variable to store all clusters
    cluster_proto=[]
    for index, row in data.iterrows():

        list_vals=()
        for idx_cols in range(len_columns):
            list_vals=list_vals+(row[idx_cols],)
        cluster_proto .append([list_vals])

    return cluster_proto,data


def agglomeration(cluster_proto):
    '''
    function to perform agglomeration
    :param cluster_proto: all clusters
    :return: final clusters and a list of sizes of last smallest merged cluster
    '''
    last_merged_cluster_size=[]
    while len(cluster_proto) > 6:
        min_dist = float('inf')
        min_i_index = float('inf')
        min_j_index = float('inf')
        for cluster1_idx in range(len(cluster_proto) - 1):
            for cluster2_idx in range(cluster1_idx + 1, len(cluster_proto)):
                dist = getDistance(cluster_proto[cluster1_idx],
                                   cluster_proto[cluster2_idx])

                if min_dist > dist:
                    min_dist = dist
                    min_i_index = cluster1_idx
                    min_j_index = cluster2_idx

        if(len(cluster_proto[min_j_index])>len(cluster_proto[min_i_index])):
            last_merged_cluster_size.append(len(cluster_proto[min_j_index]))
        else: last_merged_cluster_size.append(len(cluster_proto[min_i_index]))
        mergeCluster(cluster_proto, min_i_index, min_j_index)
        print(len(cluster_proto))
    return cluster_proto,last_merged_cluster_size

def mergeCluster(all_clusters, cluster_1, cluster_2):
    """
    Function to merge data points into clusters
    :param all_clusters: list of lists containing tuples of latitude and longitude
    :param cluster_1:    index of cluster for which the new found data point has to be added
    :param cluster_2:    index of new found data point
    """
    # Adding the data point to the cluster after finding the minimum
    # euclidean  distance
    all_clusters[cluster_1].extend(all_clusters[cluster_2])
    # Removing the data point whihc was added to the other cluster
    all_clusters.pop(cluster_2)

def getDistance(cluster_1,cluster_2):
    '''
    Function to compute euclidean distance between two clusters
    :param cluster_1:
    :param cluster_2:
    :return: distance between two clusters
    '''

    cluster1=numpy.mean(cluster_1, axis=0)
    cluster2 = numpy.mean(cluster_2, axis=0)
    dist = float('inf')
    dist= distance.euclidean(cluster1[1:], cluster2[1:])

    return dist




def main():
    '''
    Main Function to call read ,agglomeration clustering and plot dendogram functions.
    '''
    cluster_proto,data=read_data()
    plotDendogram(data)
    cluster_proto,last_merged_cluster_size=agglomeration(cluster_proto)
    print("Sizes of last merged clusters ",last_merged_cluster_size[-9:])
    print("last_merged_cluster_size",last_merged_cluster_size)
    tg=[]
    for t in cluster_proto:
        tg.append(len(t))
    print(sum(tg),tg)
    with open('output.txt', 'w') as f:
        for item in cluster_proto:
            f.write("%s\n" % item)
    f.close()


def plotDendogram(data):
    '''
    Function to plot dendogram
    :param data:
    :return:
    '''
    points = sci.linkage(data.head(50), method='average')
    sci.dendrogram(points, truncate_mode='lastp', p=50)
    plt.savefig('Dendogram.png')
    plt.show()




if __name__=='__main__':

   main()





