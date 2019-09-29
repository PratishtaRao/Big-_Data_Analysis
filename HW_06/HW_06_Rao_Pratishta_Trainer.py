"""
Title: HW_06_Rao_Pratishta_Trainer.py
Course: CSCI 720
Date: 03/15/2019
Author: Pratishta Prakash Rao
Description: Code to build decision tree for the given data
"""

import pandas as p
import sys
import math
import re
import textwrap


class Node:
    __slots__ = "data", "left", "right"

    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data


def bhattacharya_coefficients(df, column_name, class_variable):
    """
    Function to calculate the bhattacharya co-efficients
    :param df: data frame with data
    :param column_name: name of the column
    :param class_variable: taget variable
    :return: bhattacharya coeffficients for two nodes
    """

    # Splitting the attribute into two nodes

    # Count when column_name = 1 and target variable = 1
    node_a1 = len(df.loc[(df[column_name] == 1) & (df[class_variable] == 1)])

    # Count when column_name = 1 and target variable = 0
    node_a2 = len(df.loc[(df[column_name] == 1) & (df[class_variable] == 0)])

    # Count when column_name = 0 and target variable = 1
    node_b1 = len(df.loc[(df[column_name] == 0) & (df[class_variable] == 1)])

    # Count when column_name = 0and target variable = 0
    node_b2 = len(df.loc[(df[column_name] == 0) & (df[class_variable] == 0)])

    # Calculating the total count in each node
    total_a = node_a1 + node_a2
    total_b = node_b1 + node_b2

    # Calculating the class probabilities
    # Probabilities for node a
    if total_a == 0:
        p_a1, p_a2 = 0, 0
    else:
        p_a1 = node_a1 /(node_a1 + node_a2)
        p_a2 = node_a2 /(node_a1 + node_a2)
    # Probabilities for node b
    if total_b == 0:
        p_b1, p_b2 = 0, 0
    else:
        p_b1 = node_b1 /(node_b1 + node_b2)
        p_b2 = node_b2 /(node_b1 + node_b2)

    b_c = math.sqrt(p_a1 * p_b1) + math.sqrt(p_a2 * p_b2)
    return b_c


def drop_insignificant_cols(df, target_variable):
    """
    Function to drop insginificant columns
    :param df: dataframe with data
    :param target_variable: target variable
    :return: the data frame
    """
    # Column names to a list
    column_names = df.columns.values.tolist()
    # For each column if the bhattacharya co-efficients is 0
    # then drop the column from the data frame
    for name in column_names:
        if name != 'CookieIsCrumpets':
            temp_bc = bhattacharya_coefficients(df, name, target_variable)
            if temp_bc == 0.0:
                df = df.drop([name], axis=1)
    return df


def find_best_attribute(df):
    """
    Function to find the best attribute based on bhattacharyya co-efficients
    :param df: dataframe with data
    :return: the attribute with lowest bhattacharyya co-efficients
    """
    best_attribute = ''
    best_b_c = float('inf')
    target_variable = 'CookieIsCrumpets'
    column_names = df.columns.values.tolist()

    for name in column_names:
        if name != 'CookieIsCrumpets':
            b_c_value = bhattacharya_coefficients(df, name, target_variable)
            if b_c_value < best_b_c:
                best_b_c = b_c_value
                best_attribute = name
    return best_attribute


def build_decision_tree(df, initial_depth, max_depth):
    """
    Function to build a decision tree.
    :param df: the data frame with data
    :param initial_depth: initial depth of the tree
    :param max_depth: maximum depth of the tree
    :return: node with obtained decisions
    """
    # Count of the number of 1s in target variable
    yes_count = (len(df.loc[(df['CookieIsCrumpets'] == 1)]))
    # If the count is 0, then the pure leaf node percentage is 0.0
    if yes_count == 0:
        pure_leaf_node_one = 0.0
    else:
        pure_leaf_node_one = df['CookieIsCrumpets'].value_counts(normalize=True)[1]
    # Count of the number of 0s in target variable
    no_count  = (len(df.loc[(df['CookieIsCrumpets'] == 0)]))
    # If the count is 0, then the pure leaf node percentage is 0.0
    if no_count == 0:
        check_pure_leaf_node = 0.0
    else:
        check_pure_leaf_node = df['CookieIsCrumpets'].value_counts(normalize=True)[0]
    # Stopping Criteria: Max depth reached or 95% pure leaf node
    if initial_depth == max_depth or check_pure_leaf_node > 0.95 or pure_leaf_node_one > 0.95:

        if yes_count > no_count:
            return Node("1")
        else:
            return Node("0")
    else:
        # Drop columns with bhattacharrya co-eeficients = 0
        df = drop_insignificant_cols(df, 'CookieIsCrumpets')
        # Find the best attribute
        best_attribute = find_best_attribute(df)
        # Add the current root with the best attribute
        current_root = Node(best_attribute)
        # Splitting the best attribute into two nodes
        mask_left = (df[best_attribute] == 0)
        df_left = df.loc[mask_left]
        df_right = df.loc[(df[best_attribute] == 1)]
        # Dropping the column with the best attribute
        drop_left = df_left.drop([best_attribute], axis=1)
        drop_right = df_right.drop([best_attribute], axis=1)
        # Recursive call for the left side of the node
        current_root.left = build_decision_tree(drop_left, initial_depth + 1, max_depth)
        # Recursive call for the right side of the node
        current_root.right = build_decision_tree(drop_right, initial_depth + 1, max_depth)
    return current_root


# Program writing functions
# Function for emit header
def emit_classifier_header(classifier_filename):
    output_code = textwrap.dedent('''
    """
    Title: HW_06_Rao_Pratishta_Trainer.py
    Course: CSCI 720
    Date: 03/15/2019
    Author: Pratishta Prakash Rao
    Description: Code to classify given data
    """
    import csv
    import pandas as p
    import sys
    import re

    test_path = sys.argv[1]
    test_data = p.read_csv(test_path)
    column_names = test_data.columns.values.tolist()
    for value in column_names:
        check_cookie = re.findall("^Cookie.*$", value)
        if check_cookie !=[] and check_cookie != ['CookieIsCrumpets']:
            test_data = test_data.drop(check_cookie, axis = 1)
    test_data = test_data.drop(['Sleep_VALUE', 'HeightVALUE', 'Shoes_VALUE'], axis=1)
    

    # Output of our classifier will be stored in this
    out = list()
    ''')
    with open(classifier_filename, 'w') as file:
        file.write(output_code)


# Function for emit decision trees
def emit_decision(classifier_filename, node):
    output_code = emit_decision_tree(1, node)
    with open(classifier_filename, 'a') as file:
        file.write("\n\ndef classify(row_data):")
        file.write(output_code)


# Function for emit classifier call
def emit_classifier_call(classifier_filename):
    output_code = textwrap.dedent('''
    \n    
    # Classifier call
    for rows in range(test_data.shape[0]):
        out.append(classify(test_data.iloc[[rows]]))


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
        for value in out:
            writer.writerow([value])
    ''')
    with open(classifier_filename, 'a') as file:
        file.write(output_code)


def emit_decision_tree(depth, node):
    """
    Function to print the decision tree
    :param depth: depth of the tree
    :param node: the node with the decision data
    :return: output string with decision stumps
    """
    output = ""
    indent = "    " * depth
    output += "\n" + indent + "if row_data['" + node.data + "'].values[0] == 1:"
    if node.right.data == '1':
        output += "\n" + indent + "    return 1"
    elif node.right.data == '0':
        output += "\n" + indent + "    return 0"
    else:
        output += emit_decision_tree(depth + 1, node.right)
    output += "\n" + indent + "else:"
    if node.left.data == '1':
        output += "\n" + indent + "    return 1"
    elif node.left.data == '0':
        output += "\n" + indent + "    return 0"
    else:
        output += emit_decision_tree(depth + 1, node.left)
    return output


def main():
    # Usage: python HW06.py train.csv test.csv
    train_path = sys.argv[1]
    train_df = p.read_csv(train_path)
    # Copy of the original dataframe
    train_copy = train_df.copy()
    # Columns names of the data frame to a list
    column_names = train_copy.columns.values.tolist()
    # If the column name starts with cookie and ends with anything
    # drop it from the data frame except for the target variable
    for value in column_names:
        check_cookie = re.findall("^Cookie.*$", value)
        if check_cookie != [] and check_cookie != ['CookieIsCrumpets']:
            train_copy = train_copy.drop(check_cookie, axis=1)
    # Drop columns with continuous values
    train_copy = train_copy.drop(['Sleep_VALUE', 'HeightVALUE', 'Shoes_VALUE'], axis=1)
    # Function call for decision tree
    my_tree = build_decision_tree(train_copy, 0, 3)
    # Function calls for functions that write the classifier program
    emit_classifier_header("HW_06_Rao_Pratishta_classifier.py")
    emit_decision("HW_06_Rao_Pratishta_classifier.py", my_tree)
    emit_classifier_call("HW_06_Rao_Pratishta_classifier.py")
    emit_trailer("HW_06_Rao_Pratishta_classifier.py")


if __name__ == '__main__':
    main()


