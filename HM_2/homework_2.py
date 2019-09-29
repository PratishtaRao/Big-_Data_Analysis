"""
Title: homework_2.py
Course: CSCI 720
Date: 02/11/2019
Author: Pratishta Prakash Rao
Description: Code to implement Otsu's method and regularisation
"""
import numpy as np
import pandas as p
import matplotlib.pyplot as plt


def bin(df):
    """
    Function to quantize the vehicle speeds into bins

    :param df: a dataframe with the vehicle speeds
    :return: a dataframe with a new column with binned data
    """
    # Given bin size of 2 mph
    bin_size = 2
    bin_value = []

    # Getting the vehicle speeds into a list of vehicle speeds
    data_raw = df["Speed"].tolist()

    # For every speed in the list, calculating the bin value using the formula
    for v_raw in data_raw:
        temp_value = round(v_raw/bin_size)*bin_size
        bin_value.append(temp_value)

    # Plotting the quantised speeds in a histogram and adding the labels
    # plt.hist(bin_value)
    # plt.xlabel('Quantised vehicle speeds')
    # plt.ylabel('Frequency')
    # plt.title('Plot of the Quantised Vehicle Speed')
    # plt.show()

    # Adding a column with the new quantised speeds to the dataframe
    df['Binned data'] = bin_value
    return df


def bin_test(df):
    """
    Function to quantise the vehicle speeds into bins.
    The purpose of this function was to check if the
    quatisation of the vehicle speeds worked properly.

    :param df:
    :return:
    """
    bin_size = []
    # Creating bin of size 2 mph
    for bin in range(38, 80, 2):
        bin_size .append(bin)

    # Using an inbuilt function to calculate the quantised speeds
    df['Binned data'] = p.cut(df['Speed'], bin_size)
    sample = df['Binned data'].tolist()

    # Plotting the quantised speeds in a histogram and adding the labels
    plt.hist(sample)
    plt.xlabel('Quantised vehicle speeds')
    plt.ylabel('Frequency')
    plt.title('Plot of the Quantised Vehicle Speed')
    plt.show()


def otsu_method(df, column_name):
    """
    Function to implement Otsu's method for clustering

    :param df: a dataframe with the vehicle speeds
    :param column_name: name of the column  in the dataframe with vehicle speeds
    :return: None
    """

    print('Otsu\'s method without regularisation:')

    # Getting the quantised vehicle speeds into a list
    data = df[column_name].tolist()

    # Initialising the variables to 0 and infinity accordingly
    plot_mixed_variance = []
    best_mixed_variance = float("inf")
    best_threshold = 0.0
    split_ratio = 0.0

    # For every speed in the list, split the data into two dataframes with one
    # containing the speeds less than the threshold and the other containing the
    # speeds over the threshold

    for threshold in data:
        df_under = df[(df[column_name] <= threshold)]
        df_under = df_under[[column_name]]
        df_over = df[df[column_name] > threshold]
        df_over = df_over[[column_name]]

        # Calculating the weights and variance of the two dataframes
        wt_under        = float(len(df_under))/float(len(data))
        wt_over         = float(len(df_over))/float(len(data))
        variance_under  = df_under.var(ddof=0)[0]
        variance_over   = df_over.var(ddof=0)[0]

        # Calculating the mixed variance for the two dataframes
        mixed_variance  = wt_under * variance_under + wt_over * variance_over
        plot_mixed_variance.append(mixed_variance)

        # Comparing the mixed variance with the best variance, if less than the
        # best variance mixed variance is the new best variance and the best
        # threshold is the vehicle speed for this iteration
        if mixed_variance < best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = threshold

        # Calculating the split ratio of the clusters formed
            split_ratio = len(df_under) / len(df_over)

    # Sorting the data and mixed variance in order to plot the graph
    x_axis, y_axis = zip(*sorted(zip(data, plot_mixed_variance)))
    plt.plot(x_axis, y_axis)
    plt.ylabel('Mixed variance')
    plt.xlabel('Speed')
    plt.title('Speed vs Mixed Variance for Otsu\'s method')
    plt.show()

    # Print statements to print the best mixed variance, best threshold and the split ratio
    print('Minimum mixed variance:', best_mixed_variance)
    print('Best threshold:', best_threshold)
    print('Split ratio:', split_ratio)
    print('--------------------------------------------------------------------------')


def otsu_method_regularisation(df, column_name):
    """
    Function to implement Otsu's method for clustering with regularisation

    :param df: a dataframe with the vehicle speeds
    :param column_name: name of the column in the dataframe with vehicle speeds
    :return:
    """

    print('\n\nOtsu\'s method with regularisation:')
    plot_cost_function = []
    # Getting the quantised vehicle speeds into a list
    data = df[column_name].tolist()

    # Different values for alpha that needs to be plugged  in the formula to find regularisation
    alpha = [1, 1/5, 1/10, 1/20, 1/21, 1/22, 1/23, 1/24, 1/25, 1/50, 1/100, 1/1000]

    # For every value of alpha in the list calculate the best threshold
    for temp_alpha in alpha:

        # Initialising the variables to 0 and infinity accordingly
        best_cost_function = float("inf")
        best_threshold = 0.0
        cost_function = 0.0
        regularisation = 0.0
        norm_fact = 50
        split_ratio = 0.0

        # Finding the best speed to solit the data into two
        # clusters by iterating through every single speed in the list
        for threshold in data:

            # Splitting  the data points into two data frames, one with
            # values above the threshold and one with values below the threshold
            df_under = df[(df[column_name] <= threshold)]
            df_under =  df_under[[column_name]]
            df_over = df[df[column_name] > threshold]
            df_over = df_over[[column_name]]

            # Calculating the weights and variance for the two dataframes
            wt_under = float(len(df_under)) / float(len(data))
            wt_over = float(len(df_over)) / float(len(data))
            variance_under = df_under.var(ddof=0)[0]
            variance_over = df_over.var(ddof=0)[0]

            # Calculating the mixed variance for the two dataframes
            mixed_variance = wt_under * variance_under + wt_over * variance_over

            # Calculating the regularisation with the given alpha
            regularisation = abs((len(df_under))-len(df_over))/(norm_fact * temp_alpha)

            # Cost function is equal to the sum of mixed variance which is the objective
            # function and the regularisation
            cost_function = mixed_variance + regularisation
            plot_cost_function.append(cost_function)


            # Comparing the cost function with the best cost function, if less than the
            # best cost function, cost function is the new best variance and the best
            # threshold is the vehicle speed for this iteration
            if cost_function < best_cost_function:
                best_cost_function = cost_function
                best_threshold = threshold
                split_ratio = len(df_under)/len(df_over)

        # Print statements to print the best cost function, best threshold and the split ratio
        print(' Value of alpha:', temp_alpha, ' \nBest threshold:', best_threshold)
        print('Split ratio:', split_ratio)
        print('Best Cost Function:', best_cost_function)
        print('--------------------------------------------------------------------------')

    # Sorting the data and mixed variance in order to plot the graph
    x_axis, y_axis = zip(*sorted(zip(data, plot_cost_function)))
    plt.plot(x_axis, y_axis)
    plt.ylabel('Cost Function')
    plt.xlabel('Speed')
    plt.title('Speed vs Cost Function for Otsu\'s method with regularisation')
    plt.show()


def mystery_data():
    """
    Function to find the descriptive statics for the given
    mystery data file
    :return:
    """
    # Getting the data from a csv file to a dataframe using pandas
    data = p.read_csv("/Users/pratishtarao/PycharmProjects/Bigdata/Mystery_Data.csv")
    m_unit = data['Measures (Mystery Units)']

    # Calculating the descriptive statistics before the removal of last element
    mean_m_unit          = m_unit.mean()
    median_m_unit        = m_unit.median()
    mode_m_unit          = m_unit.mode()
    variance_m_unit      = m_unit.var(ddof=0)
    std_deviation_m_unit = m_unit.std(ddof=0)
    max_value = m_unit.max()
    min_value = m_unit.min()
    mid_range = float((max_value + min_value)/2)

    # Printing the  descriptive statistics found from the above formula
    print('--------------------------------------------------------------------------')
    print('Descriptive statistics for Mystery data:')
    print('\n','Mean:', mean_m_unit,'\n', 'Median:',
          median_m_unit, '\n', 'Mode:', mode_m_unit, '\n',
          'Variance:', variance_m_unit, '\n',
          'Standard Deviation:', std_deviation_m_unit,
          '\n', 'Mid Range:', mid_range)

    # Removing the last element in the list
    new_m_unit = m_unit[:-1]

    # Calculating the descriptive statistics after the removal of the last element
    mean_unit = new_m_unit.mean()
    median_unit = new_m_unit.median()
    mode_unit = new_m_unit.mode()
    variance_unit = new_m_unit.var(ddof=0)
    std_deviation_unit = new_m_unit.std(ddof=0)
    maximum = new_m_unit.max()
    minimum = new_m_unit.min()
    new_mid_range = float((maximum + minimum)/2)

    # Printing the  descriptive statistics found from the above formula
    # after removing the last element in the list
    print('--------------------------------------------------------------------------')
    print('Descriptive statistics for Mystery data after removing the last value:')
    print('\n', 'Mean:', mean_unit, '\n', 'Median:',
          median_unit, '\n', 'Mode:', mode_unit, '\n',
          'Variance:', variance_unit, '\n',
          'Standard Deviation:', std_deviation_unit,
          '\n', 'Mid Range:', new_mid_range)

    # Using Otsu's method to split the data into two groups
    print('--------------------------------------------------------------------------')
    print('Otsus method of clustering for mystery data:\n')
    otsu_method(data, 'Measures (Mystery Units)')


def main():
    df = p.read_csv("/Users/pratishtarao/PycharmProjects/Bigdata/DATA_v2185f_FOR_CLUSTERING_using_Otsu.csv")
    value = bin(df)
    print('Ostus method for clustering vehicle speeds:\n')
    otsu_method(value, 'Binned data')
    otsu_method_regularisation(value, 'Binned data')
    mystery_data()


if __name__ == '__main__':
    main()
