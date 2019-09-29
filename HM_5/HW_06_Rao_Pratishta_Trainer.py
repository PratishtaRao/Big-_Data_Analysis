"""
Title: HW_06_Rao_Pratishta_Trainer.py
Course: CSCI 720
Date: 03/03/2019
Author: Pratishta Prakash Rao
Description: Code to build decision stumps for the given data
"""

import pandas as p
import math
import sys
import textwrap


def log(x):
    """
    Function to get the log of a number
    :param x: a number
    :return: log of the number to the base 2 if x is not 0
    """

    if x == 0:
        return 0
    else:
        return math.log(x, 2)


def data_pre_processing(continuous_values, df):
    # Pre processing the attributes with  continuous values
    # Using Otsu's method finding the best threshold to split the data into 0s and 1s
    for column_name in continuous_values:
        data = df[column_name].tolist()

        # Initialising the variables to 0 and infinity accordingly
        best_mixed_variance = float("inf")
        best_threshold = 0.0

        # For every data point in the column calculate the best threshold
        #  for the split

        for threshold in data:
            df_under = df[(df[column_name] <= threshold)]
            df_under = df_under[[column_name]]
            df_over = df[df[column_name] > threshold]
            df_over = df_over[[column_name]]

            # Calculating the weights and variance of the two dataframes
            wt_under = float(len(df_under)) / float(len(data))
            wt_over = float(len(df_over)) / float(len(data))
            variance_under = df_under.var(ddof=0)[0]
            variance_over = df_over.var(ddof=0)[0]

            # Calculating the mixed variance for the two dataframes
            mixed_variance = wt_under * variance_under + wt_over * variance_over

            # Comparing the mixed variance with the best variance, if less than the
            # best variance mixed variance is the new best variance and the best
            # threshold is the vehicle speed for this iteration
            if mixed_variance < best_mixed_variance:
                best_mixed_variance = mixed_variance
                best_threshold = threshold

        # Set all data points lesser than and equal to threshold as 0
        mask_zero = df[column_name] <= best_threshold
        df.loc[mask_zero, column_name] = 0

        # Set all data points greater than threshold as 1
        mask_one = df[column_name] != 0
        df.loc[mask_one, column_name] = 1

        # Converting the column values to have datatype int
        df[column_name] = df[column_name].astype(int)


def find_best_attribute(df):
    """
    Function to find the attribute with minimum mixed entropy
    :param df: the data frame that consists of the data to find best attribute
    """

    # Initialising the variables
    target_variable = 'CookieIsChocKrinkle'
    minimum_entropy = float("inf")
    best_attribute = ''

    # Getting the column names into a list
    column_names = list(df.columns.values)

    # Removing the Target varibale from the list
    column_names.remove('CookieIsChocKrinkle')

    # Removing the columns with continuous values from the list
    column_names.remove('Sleep_VALUE')
    column_names.remove('HeightVALUE')
    column_names.remove('Shoes_VALUE')

    # Creating new list with column names of continuous values
    continuous_values = list()
    continuous_values.append('Sleep_VALUE')
    continuous_values.append('HeightVALUE')
    continuous_values.append('Shoes_VALUE')

    # Pre Processing the continuous value
    data_pre_processing(continuous_values, df)

    # Calculating the minimum mixed entropy for the columns with binary value
    for col_value in column_names:
        mixed_entropy = find_mixed_entropy(df, col_value, target_variable)
        if mixed_entropy < minimum_entropy:
            minimum_entropy = mixed_entropy
            best_attribute = col_value
    print('Best attribute without the attributes with continuous values:', best_attribute)

    # If the length of sys.argv is more than 1 then include the attributes that had
    # continuous values to  find the best attribute along with the rest of the column names

    if len(sys.argv) > 3:
        for continuous_val in continuous_values:
            continuous_entropy = find_mixed_entropy(df, continuous_val, target_variable)
            if continuous_entropy < minimum_entropy:
                minimum_entropy = continuous_entropy
                best_attribute = continuous_val
        print('Best attribute including the attributes with continuous value:', best_attribute)
    print(minimum_entropy)
    return best_attribute


def find_mixed_entropy(df, column_name, target_variable):
    """
    Function to find the mixed entropy of the given column name
    :param df: data frame with the data
    :param column_name: name of the column for which mixed entropy needs to be found
    :param target_variable: target variable for the given dataset
    :return: mixed entropy that is calulated
    """
    # Splitting the attribute into two nodes
    # Count when column_name = 1 and target variable = 1

    node_one_yes_yes = len(df.loc[(df[column_name] == 1) & (df[target_variable] == 1)])

    # Count when column_name = 1 and target variable = 0
    node_one_yes_no =  len(df.loc[(df[column_name] == 1) & (df[target_variable] == 0)])

    # Count when column_name = 0 and target variable = 1
    node_two_no_yes =  len(df.loc[(df[column_name] == 0) & (df[target_variable] == 1)])

    # Count when column_name = 10and target variable = 0
    node_two_no_no =   len(df.loc[(df[column_name] == 0) & (df[target_variable] == 0)])

    # Calculating the total count in each node
    total_node_one = node_one_yes_yes + node_one_yes_no
    total_node_two = node_two_no_yes  + node_two_no_no

    # Finding the probabilities for each split
    if total_node_one == 0:
        prob_node_one_yes, prob_node_one_no = 0, 0
    else:
        prob_node_one_yes, prob_node_one_no = node_one_yes_yes / total_node_one, node_one_yes_no / total_node_one

    if total_node_two == 0:
        prob_node_two_yes, prob_node_two_no = 0, 0
    else:
        prob_node_two_yes, prob_node_two_no = node_two_no_yes / total_node_two, node_two_no_no / total_node_two

    # Calculating the entropy for each node
    entropy_node_one = - prob_node_one_yes * log(prob_node_one_yes) - prob_node_one_no * log(prob_node_one_no)
    entropy_node_two = - prob_node_two_yes * log(prob_node_two_yes) - prob_node_two_no * log(prob_node_two_no)

    # Bhattacharyya co-efficient
    bc = math.sqrt(prob_node_one_yes * prob_node_two_yes) + math.sqrt(prob_node_one_no + prob_node_two_no)

    # Calculating the mixed entropy
    mixed_entropy = (total_node_one * entropy_node_one + total_node_two * entropy_node_two)/(total_node_one + total_node_two)
    return mixed_entropy


# Program writing functions
# Function for emit header
def emit_classifier_header(classifier_filename, best_attr):

    output_code =textwrap.dedent('''
    import csv
    import pandas as p
    import sys
    
    best_attr = \"''' + best_attr + '''\"
    test_path = sys.argv[1]
    test_data = p.read_csv(test_path)
    
    # Convert attribute of interest to list
    attr_vals = test_data[best_attr].tolist()
    
    # Output of our classifier will be stored in this
    out = list()
    ''')
    with open(classifier_filename, 'w') as file:
        file.write(output_code)


# Function for emit decision stump
def emit_decision(classifier_filename):
    output_code = textwrap.dedent('''
    # Function for decision stumps
    def classify(value):
        if value <= 0:
            return 0
        return 1
            ''')
    with open(classifier_filename, 'a') as file:
        file.write(output_code)


# Function for emit classifier call
def emit_classifier_call(classifier_filename):
    output_code = textwrap.dedent('''
    # Classifier call
    for val in attr_vals:
        out.append(classify(val))

    ''')
    with open(classifier_filename, 'a') as file:
        file.write(output_code)


# Function for emit trailer
def emit_trailer(classifier_filename):
    output_code = textwrap.dedent('''
    # Write output list to a csv file
    with open('HW_06_Rao_Pratishta_My_Classification.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Result'])
        for v in out:
            writer.writerow([v])
    ''')
    with open(classifier_filename, 'a') as file:
        file.write(output_code)


def main():
    # Usage: python HW**.py train.csv test.csv continuous_vars_flag=<1,0>
    # continuous_vars_flag is created in order to choose to evaulte attributes with
    # continuous values or not
    train_path = sys.argv[1]
    train_df = p.read_csv(train_path)
    train_copy = train_df.copy()
    best_attr = find_best_attribute(train_copy)
    # Function calls for creating the classifier program
    emit_classifier_header("HW_06_Rao_Pratishta_classifier.py", best_attr)
    emit_decision("HW_06_Rao_Pratishta_classifier.py")
    emit_classifier_call("HW_06_Rao_Pratishta_classifier.py")
    emit_trailer("HW_06_Rao_Pratishta_classifier.py")


if __name__ == '__main__':
    main()