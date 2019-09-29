"""
Title: homework_3.py
Course: CSCI 720
Date: 02/17/2019
Author: Pratishta Prakash Rao
Description: Code to build a One Dimensional classifier
"""
import pandas as p
import matplotlib.pyplot as plt
import math
import numpy as np


class Cost_function:

    def __init__(self, w1=1, w2=1):
        self.w1=w1
        self.w2=w2

    def cost_function(self, df_under, df_over):
        # Calculating the false alarms and misses
        num_misses = (df_under['Aggressive'] == 1).sum()
        num_false_alarms = (df_over['Aggressive'] == 0).sum()
        n_wrong = self.w1 * num_false_alarms + self.w2 * num_misses
        return n_wrong


def bin(df):
    """
    Function to quantize the vehicle speeds into bins

    :param df: a dataframe with the vehicle speeds
    :return: a dataframe with a new column with binned data
    """
    # Given bin size of 0.5 mph
    bin_size = 0.5
    bin_value = []

    # Getting the vehicle speeds into a list of vehicle speeds
    data_raw = df["Speed"].tolist()

    # For every speed in the list, calculating the bin value using the formula
    for v_raw in data_raw:
        temp_value = round(v_raw/bin_size)*bin_size
        bin_value.append(temp_value)

    # Adding a column with the new quantised speeds to the data frame
    df['Binned data'] = bin_value
    return df


def classification(df, cf):
    """
    Function to classify the given data points and plotting
    the ROC curve

    :param df: data frame with the data points
    :param cf: the cost function that needs to be used
    """
    best_misclass_rate = float('inf')
    plot_cost_function = []
    tpr = []
    fpr = []

    # Creating the data frame with only two columns that are needed
    df = df[['Binned data','Aggressive']]

    # Sorting the data frame with column Binned data in ascending order of speeds
    df = df.sort_values(by= ['Binned data'], ascending=True)

    # Getting the quantised vehicle speeds into a list
    data = df['Binned data'].tolist()
    best_threshold = 0.0

    # Finding all possible speed thresholds
    minimum_speed = min(data)
    # Floor of the minimum speed
    minimum_speed = math.floor(minimum_speed)
    maximum_speed = max(data)
    # Ceil of the minimum speed
    maximum_speed = math.ceil(maximum_speed)
    all_possible_thresholds = []
    for i in np.arange(minimum_speed, maximum_speed, 0.5):
        all_possible_thresholds.append(i)

    # For every speed in the list, split the data into two dataframes with one
    # containing the speeds less than the threshold and the other containing the
    # speeds over the threshold
    for threshold in all_possible_thresholds:
        data_under = df[(df['Binned data'] <= threshold)]
        data_over =  df[(df['Binned data'] > threshold)]

        # Calling the cost function by passing the two groups of speed
        num_wrong = cf(data_under, data_over)
        plot_cost_function.append(num_wrong)

        if num_wrong < best_misclass_rate:
            best_misclass_rate = num_wrong
            best_threshold = threshold

        # Calculating the True Positve Rate and the False Positive Rate
        fp = (data_over['Aggressive'] == 0).sum()
        fn = (data_under['Aggressive'] == 1).sum()
        tp = (data_over['Aggressive'] == 1).sum()
        tn = ((data_under['Aggressive'] == 0).sum())

        temp_tpr = tp / (tp + fn)
        temp_fpr = fp / (fp + tn)
        tpr.append(temp_tpr)
        fpr.append(temp_fpr)

    # Calculating the # of misses and # of false alarms for the best threshold
    temp_under = df[(df['Binned data'] <= best_threshold)]
    temp_over = df[(df['Binned data'] > best_threshold)]
    misses = (temp_under['Aggressive'] == 1).sum()
    false_alarms = (temp_over['Aggressive'] == 0).sum()
    print('Best threshold:',best_threshold)
    # print('Best missclassification rate:',best_misclass_rate)
    print('number aggressive drivers let go(misses) :',misses)
    print('number of non aggressive pulled over(false alarms):', false_alarms)

    # Plotting the speed vs cost function
    # Sorting the data in order to plot
    x_axis, y_axis = zip(*sorted(zip(all_possible_thresholds, plot_cost_function)))
    # Plot for speed vs cost function
    plt.figure()
    plt.plot(x_axis, y_axis)
    # Labelling the x axis, y axis and title
    plt.ylabel('Cost Function')
    plt.xlabel('Speed')
    plt.title('Speed vs Cost Function')
    plt.show()

    # Plotting ROC curve
    # Sorting the data in order to plot
    x_axis, y_axis = zip(*sorted(zip(fpr, tpr)))
    # Plot for speed vs cost function
    ax = plt.axes()
    # Plotting the coin toss line
    ax.plot([0, 1], [0, 1],'--', transform=ax.transAxes)
    # to make the plot with axes as square
    plt.gca().set_aspect('equal', adjustable='box')
    # Plotting the fpr, tpr
    plt.plot(x_axis, y_axis, linestyle='-', marker='o')
    # Adding the legends
    plt.gca().legend(('Receiver Operating Curve', 'Coin toss line'))

    # Labelling the x axis, y axis and title
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    # Calculating the area under the curve
    auc = np.trapz(y_axis,x_axis)
    print('Area under the curve:', auc)
    print('--------------------------------------------------------------------------')


def main():
    # Reading the csv file into a data frame
    df = p.read_csv('/Users/pratishtarao/PycharmProjects/Bigdata/data_for_homework_3.csv')
    # Quantizing the data
    value = bin(df)
    print('--------------------------------------------------------------------------')
    print('cost function = (# of false alarms + # of misses)')
    classification(value, Cost_function().cost_function)
    print('cost function = (2 times # of false alarms + # of misses)')
    classification(value, Cost_function(w1=2, w2=1).cost_function)
    print('cost function = ( # of false alarms + 2 times # of misses)')
    classification(value, Cost_function(w1=1, w2=2).cost_function)


if __name__ == '__main__':
    main()