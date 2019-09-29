
import csv
import pandas as p
import sys

best_attr = "BrkDnkIsCoffee"
test_path = sys.argv[1]
test_data = p.read_csv(test_path)

# Convert attribute of interest to list
attr_vals = test_data[best_attr].tolist()

# Output of our classifier will be stored in this
out = list()

# Function for decision stumps

def classify(value):
    if value <= 0:
        return 0
    return 1

# Classifier call
for val in attr_vals:
    out.append(classify(val))


# Write output list to a csv file
with open('HW_06_Rao_Pratishta_My_Classification.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['Result'])
    for v in out:
        writer.writerow([v])
