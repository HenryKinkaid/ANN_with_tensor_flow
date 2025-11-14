import tensorflow as tf
import csv
import numpy as np
from ANN_File import ANN
MAX_X = 1626
MAX_Y = 988
import random
import matplotlib.pyplot as plt
USE_SAVED_MODEL = True

def main():
    data = extract_data('ball_positions.csv')
    train, test = simple_train_test_split(data)

    X_train, y_train = prepare_X_y(train)
    X_test, y_test   = prepare_X_y(test)
    # print(data)
    if not USE_SAVED_MODEL:
        model = ANN()

        history = model.train_model_n_epochs(
            n=50,  # number of epochs
            batch_size=32,
            inputs=X_train,
            outputs=y_train
        )
    else:
        model = ANN(filepath="my_model.keras")

    predictions = model.test_model(X_test)
    print("First 10 predictions vs actual:")
    print(np.hstack([predictions[:10], y_test[:10]]))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='Actual Path', s=20)
    plt.scatter(predictions[:, 0], predictions[:, 1], color='red', label='Predicted Path', s=20, marker='x')
    plt.title("Predicted vs Actual Ball Path")
    plt.xlabel("X Position (normalized)")
    plt.ylabel("Y Position (normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()

    diff = predictions - y_test
    err = np.sqrt(np.sum(np.square(diff), -1)/2) # RMSE
    max_error = np.argmax(err)
    print("inputs vs outputs for the max error: ")
    print(f"inputs: {X_test[max_error]}")
    print(f"expected output: {y_test[max_error]}")
    print(f"actual output: {predictions[max_error]}")
    print(f"error: {err[max_error]}")

    if not USE_SAVED_MODEL and input("Save this model? (y/n) ").lower() == "y":
        model.model.save(input("file name? ") + ".keras")







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
                #skip bad rows that aren't properly formatted (shouldn't be in there but just in case)
                continue

    delta_x = 100/MAX_X # max distance for two positions that are in the same group
    delta_sq = delta_x ** 2 # writing this now since will ref later
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
            # print(f"row num: {len(run_sets)} row: {run}")

    return run_sets

def simple_train_test_split(data, test_size=0.2, random_seed=42):
    """Splits X and y into training and testing sets"""
    random.seed(random_seed)
    random.shuffle(data)
    split = int(len(data) * (1 - test_size))
    train_sets = data[:split]
    test_sets = data[split:]

    return np.array(train_sets), np.array(test_sets)

def prepare_X_y(run_sets):
    inputs = []
    labels = []

    for run in run_sets:
        # flatten first 3 points for input
        inp = [coord for point in run[:3] for coord in point[:2]]  # x and y only
        out = run[3][:2]  # x and y of 4th point
        inputs.append(inp)
        labels.append(out)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)

if __name__ == "__main__":
    main()