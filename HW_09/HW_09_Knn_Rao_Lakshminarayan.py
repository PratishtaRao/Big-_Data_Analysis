'''
Program to classify each of the 20 shoppers as one of the six cluster prototypes
using 1NN and euclidean distance
'''
import pandas
from scipy.spatial import distance
import numpy

def main():
    '''
    Function to read in classifier file ,compute center of each cluster and output of agglomeration clustering
    :return:
    '''

    # variable to store all clusters after agglomeration.
    all_cluster=[]

    # read in classifier file
    data = pandas.read_csv('/Users/srinivaslakshminarayan/PycharmProjects/bda/HW09_CLASSIFY_THESE_2185.csv',
                           header=None)
    len_cols=len(data.columns)

    # populate all_cluster by reading from output file
    with open("/Users/srinivaslakshminarayan/PycharmProjects/bda/output.txt") as fp:
        for line in fp:
            list_each_cluster=line.split("(")
            each_line=[]

            for each_val in list_each_cluster:
                each_val=each_val.split(',')
                if(len(each_val)>=len_cols):
                    list_vals = ()
                    for val in each_val:
                        if(val!=' ' and ')' not in val):
                            list_vals = list_vals +(int(val),)
                        elif(val!='' and ')'  in val):
                            val=val.split(')')
                            v=val[0]
                            v=v.strip()
                            list_vals = list_vals + (int(v),)
                    each_line.append(list_vals)

            print(len(each_line))
            all_cluster.append(each_line)
    computeClusterPrototype(all_cluster)
    assignClusters(data,all_cluster)



def computeClusterPrototype(all_cluster):
    '''
    Function to compute the centers of each clusters and print them
    :param all_cluster:
    :return:
    '''
    for each_cluster in all_cluster:
        print("Cluster Centers are",numpy.mean(each_cluster[1:],axis=0))


def assignClusters(data,all_cluster):
    '''
    Function to calculate euclidean distance between each data point from all clusters and
    put classifier data point into the cluster.(assign clusters to each data point in classifier file)
    :param data: csv file data for classifier file
    :param all_cluster: all agglomerated clusters
    :return:
    '''

    shoppers = ['Vegan', 'Kosher', 'Weed eater', 'Hillbilly', 'Family', 'Other']
    for index,rows in data.iterrows():
        row_data=[]
        for each_value in rows:
            row_data.append(each_value)
        cluster_index = 0
        min_index = float('inf')
        min_dist = float('inf')
        for each_clusters in all_cluster:
            for cluster in each_clusters:
                dist=distance.euclidean(row_data[1:], cluster[1:])
                if(min_dist>dist):
                    min_dist=dist
                    min_index=cluster_index
            cluster_index += 1
        print("Shopper ",index+1," Goes Into Cluster ",shoppers[min_index])
        all_cluster[min_index].append(row_data)
        # print(index)

if __name__=='__main__':

   main()