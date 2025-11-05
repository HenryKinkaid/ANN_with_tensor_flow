import tensorflow as tf
import csv

def main():
    ## where the data is held
    data_set = []

    with open('ball_positions.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            # creates the data and puts it in the data_set
            data_set.append(list(map(int,row)))

    delta_x = 100
    run_sets = []
    for data in range(len(data_set)):
        next_run = True
        if data+3 >= len(data_set):
            break
        for next in range(4):
            diff_x_pos = data_set[data][1] - data_set[data+next][1]
            diff_y_pos = data_set[data][2] - data_set[data+next][2]
            if delta_x**2 < (diff_x_pos**2 + diff_y_pos**2):
                next_run = False
        if next_run:
            run_sets.append([data_set[data][1:], data_set[data+1][1:], data_set[data+2][1:], data_set[data+3][1:]])
            print('row num: ' + str(data) + ' row: ' + str(run_sets[-1]))





if __name__ == "__main__":
    main()