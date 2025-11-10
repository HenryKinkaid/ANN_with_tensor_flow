import tensorflow as tf
import csv
import numpy as np
from ANN_File import ANN
MAX_X = 1626
MAX_Y = 988
import random

def main():
    data = extract_data('ball_positions.csv')
    train, test = simple_train_test_split(data)
    # print(data)
    model = ANN()

    model.train_model_n_epochs(1,1,train[:,:3],train[:,3:])
    print(model.test_model(test[:,:3]))

def extract_data(file):
    """Import Data from CSV file and identify runs"""
    data_set = []

    with open(file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                #convert data from pixel number to a number from 0.0-> 1.0 as prep for ANN
                values = list(map(int, row))
                x = values[1] / MAX_X
                y = values[2] / MAX_Y
                data_set.append([x, y] + values[3:])
            except (ValueError, IndexError):
                #skip bad rows that aren't properly formated (shouldn't be in there but just in case)
                continue

    delta_x = 100/MAX_X # max distance for two positions that are in the same group
    delta_sq = delta_x ** 2 #writing this now since will ref later
    run_sets = []

    for i in range(len(data_set) - 3):
        valid_run = True
        for j in range(1, 4):
            dx = data_set[i + j - 1][0] - data_set[i + j][0]
            dy = data_set[i + j - 1][1] - data_set[i + j][1]
            if dx * dx + dy * dy > delta_sq:
                valid_run = False
                break

        if valid_run:
            run = data_set[i:i + 4]
            run_sets.append(run)

    return run_sets

def simple_train_test_split(data, test_size=0.2, random_seed=42):
    """Splits X and y into training and testing sets"""
    random.seed(random_seed)
    random.shuffle(data)
    split = int(len(data) * (1 - test_size))
    train_sets = data[:split]
    test_sets = data[split:]

    return np.array(train_sets), np.array(test_sets)

if __name__ == "__main__":
    main()