
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


def classify(row_data):
    if row_data['BrkDnkIsTea'].values[0] == 1:
        if row_data['PizTopIsPepperoni'].values[0] == 1:
            return 0
        else:
            if row_data['DinDnkIsPapaya'].values[0] == 1:
                return 0
            else:
                return 1
    else:
        return 0


# Classifier call
for rows in range(test_data.shape[0]):
    out.append(classify(test_data.iloc[[rows]]))



# Write output list to a csv file
with open('HW_06_Rao_Pratishta_My_Classification.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['Result'])
    for value in out:
        writer.writerow([value])
