import tensorflow as tf
import csv
import numpy as np
MAX_X = 1626
MAX_Y = 988

def main():
    data = extract_data('ball_positions.csv')
    # print(data)

def extract_data(file):
    ## where the data is held
    data_set = []

    with open(file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            # creates the data and puts it in the data_set
            temp_row_list = list(map(int, row))[1:]
            temp_row_list[0] /= MAX_X
            temp_row_list[1] /= MAX_Y
            data_set.append(temp_row_list)
        # max = 0
        # for row in data_set:
        #     if row[2] > max:
        #         max = row[2]
        #
        # print(max)

    delta_x = 100/MAX_X # max distance for two positions that are in the same group
    run_sets = []

    for data in range(len(data_set)):
        next_run = True
        if data + 3 >= len(data_set):
            break
        for next in range(1, 4):
            diff_x_pos = data_set[data + next - 1][0] - data_set[data + next][0]
            diff_y_pos = data_set[data + next - 1][1] - data_set[data + next][1]
            if delta_x ** 2 < (diff_x_pos ** 2 + diff_y_pos ** 2):
                next_run = False

        if next_run:
            run_sets.append(
                [data_set[data], data_set[data + 1], data_set[data + 2], data_set[data + 3]])
            print('row num: ' + str(len(run_sets)) + ' row: ' + str(run_sets[-1]))

    return run_sets

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