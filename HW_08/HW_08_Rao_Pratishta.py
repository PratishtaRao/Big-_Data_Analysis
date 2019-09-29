"""
Title: HW_08_Rao_Pratishta.py
Course: CSCI 720
Date: 03/31/2019
Author: Pratishta Prakash Rao, Srikanth Lakshminarayan
Description: Code to implement the agglomeration clustering
"""


from haversine import haversine
from geopy.geocoders import Nominatim
import pandas
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point
import scipy.cluster.hierarchy as sci


def mergeCluster(all_clusters, cluster_1, cluster_2):
    """
    Function to merge data points into clusters
    :param all_clusters: list of lists containing tuples of latitude and longitude
    :param cluster_1:    index of cluster for which the new found data point has to be added
    :param cluster_2:    index of new found data point
    """
    # Adding the data point to the cluster after finding the minimum
    # haversine distance
    all_clusters[cluster_1].extend(all_clusters[cluster_2])
    # Removing the data point whihc was added to the other cluster
    all_clusters.pop(cluster_2)


def getDistance(list_i_cluster, list_j_cluster):
    """
    Function to get the single linkage using haversine distance
    :param list_i_cluster: list of data points
    :param list_j_cluster: list of data points
    :return:               minimum distance
    """

    min_dist = float('inf')

    # For each data point in a cluster find the
    # haversine distance with each data point in
    # the other cluster
    for each_i in list_i_cluster:

        for each_j in list_j_cluster:
            dist = haversine(each_i, each_j)

            if (min_dist > dist):
                min_dist = dist

    return min_dist


def agglomeration(all_cluster_latitude_longitude):
    """
    Function to implement the agglomeration clustering
    :param all_cluster_latitude_longitude: list containin the latitude and longitude of the city
    :return:
    """

    while len(all_cluster_latitude_longitude) > 12:
        min_dist = float('inf')
        min_i_index = float('inf')
        min_j_index = float('inf')
        for cluster1_idx in range(len(all_cluster_latitude_longitude) - 1):

            for cluster2_idx in range(cluster1_idx + 1, len(all_cluster_latitude_longitude)):
                dist = getDistance(all_cluster_latitude_longitude[cluster1_idx],
                                   all_cluster_latitude_longitude[cluster2_idx])

                if min_dist > dist:
                    min_dist = dist
                    min_i_index = cluster1_idx
                    min_j_index = cluster2_idx

        mergeCluster(all_cluster_latitude_longitude, min_i_index, min_j_index)

    print("Process complete")

    total_count_in_cluster = []
    for h in all_cluster_latitude_longitude:
        total_count_in_cluster.append(len(h))
        print(len(h))

    print("Total count in cluster", sum(total_count_in_cluster))
    return all_cluster_latitude_longitude


def get_gps_points(df):
    """
    Function to get the latitude and longitude of the cities
    :param df:  data frame with cities
    :return:    list of litst containing latitude and longitude
    """
    geolocator = Nominatim()
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    rows, columns = df.shape
    my_data = df.values
    all_cluster_latitude_longitude=[]

    try:
        with open('lat_lon1.csv', mode='w') as myfile:
            if rows is not None:
                for rows in my_data:
                    location_obj = geolocator.geocode(rows)
                    myfile.write(str(location_obj.latitude)+","+str(location_obj.longitude))
                    myfile.write('\n')
                    all_cluster_latitude_longitude.append([(location_obj.longitude, location_obj.latitude)])
    except:
        print("An error occured while getting values for latitudes and longitudes.Reading from lat_long.csv. "
              "Pls make sure this python file and csv files are kept together.")

    all_cluster_latitude_longitude = []


    """
    The points which we were getting from api were sometimes wrong and were not consistent.
    hence a file is made from which all points are read.
    """
    all_cluster_latitude_longitude=[]
    dt = pandas.read_csv('lat_long.csv', header=None)
    my_data = dt.values
    for dat in my_data:
            all_cluster_latitude_longitude.append([(float(dat[1]), float(dat[0]))])

    return all_cluster_latitude_longitude




def plotting(all_cluster_latitude_longitude,list_colors):
    '''
    Method to plot the given clusters onto the world map.
    :param all_cluster_latitude_longitude: List of List of clusters
    :param list_colors: list of colours for different clusters
    :return: void
    '''
    f, ax = plt.subplots(1, figsize=(12, 6))
    ax.set_title('Clusters')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, facecolor='lightgray', edgecolor='gray')
    ax.set_ylim([-90, 90])
    ax.set_axis_off()
    plt.axis('equal')
    index = 0
    for clusters in all_cluster_latitude_longitude:
        latitude = []
        longitude = []
        for clstr in clusters:
            longitude.append(clstr[1])
            latitude.append(clstr[0])

        dataframe = pandas.DataFrame({'Latitude': latitude, 'Longitude': longitude})
        dataframe['Coordinates'] = list(zip(dataframe.Longitude, dataframe.Latitude))
        dataframe['Coordinates'] = dataframe['Coordinates'].apply(Point)
        crs = {'init': 'epsg:4326'}
        gdf = geopandas.GeoDataFrame(dataframe, crs=crs, geometry='Coordinates')
        gdf.crs

        gdf.plot(ax=ax, marker='o', color=list_colors[index], markersize=.5, linewidth="5")

        index += 1
    plt.show()



def dendogram(city_country_csv,data):
    '''
    Method to create a dendogram of the first 50 points
    :param city_country_csv: dataframe of city_country values
    :param data: dataframe of data
    :return:
    '''

    city_country_csv = city_country_csv.head(50)
    city_vals = city_country_csv['City'].tolist()
    country_vals = city_country_csv['Country'].tolist()
    labels = [str(x) + " " + str(y) for x, y in zip(city_vals, country_vals)]
    points = sci.linkage(data.head(50), method='single')
    sci.dendrogram(points, truncate_mode='lastp', p=50, labels=labels)
    plt.show()


def main():
    list_colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'magenta', 'white', 'black', 'orange', 'pink', 'purple',
                   'gray']

    # To read the city and country into a data frame
    city_country_csv = pandas.read_csv("/Users/srinivaslakshminarayan/PycharmProjects/bda/CS_720_City_Country.csv")
    all_cluster_latitude_longitude= get_gps_points(city_country_csv)

    all_cluster_latitude_longitude=agglomeration(all_cluster_latitude_longitude)
    plotting(all_cluster_latitude_longitude,list_colors)
    data = pandas.read_csv('lat_long.csv', header=None)
    dendogram(city_country_csv,data)

if __name__ == '__main__':
    main()


