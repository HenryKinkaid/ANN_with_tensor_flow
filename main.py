import tensorflow as tf
import csv

def main():
    ## where the data is held
    data_set = []

    with open('ball_positions.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            # creates the data and puts it in the data_set
            data_set.append(row)


if __name__ == "__main__":
    main()