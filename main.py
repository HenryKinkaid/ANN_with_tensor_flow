import tensorflow as tf
import csv
import numpy as np
MAX_X = 1626
MAX_Y = 988

def main():
    ## where the data is held
    data_set = []

    with open('ball_positions.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            # creates the data and puts it in the data_set
            data_set.append(list(map(int,row)))
        # max = 0
        # for row in data_set:
        #     if row[2] > max:
        #         max = row[2]
        #
        # print(max)

    delta_x = 100
    run_sets = []

    for data in range(len(data_set)):
        next_run = True
        if data+3 >= len(data_set):
            break
        for next in range(1,4):
            diff_x_pos = data_set[data+next-1][1] - data_set[data+next][1]
            diff_y_pos = data_set[data+next-1][2] - data_set[data+next][2]
            if delta_x**2 < (diff_x_pos**2 + diff_y_pos**2):
                next_run = False

        if next_run:
            run_sets.append([data_set[data][1:], data_set[data+1][1:], data_set[data+2][1:], data_set[data+3][1:]])
            # print('row num: ' + str(len(run_sets)) + ' row: ' + str(run_sets[-1]))

def simple_train_test_split(X, y, test_size=0.2, random_seed=42):
    """Splits X and y into training and testing sets"""
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_point = int(len(X) * (1 - test_size))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()