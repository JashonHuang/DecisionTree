from sklearn.model_selection import train_test_split
from DecisionTreeID3 import *
from DecisionTreeC45 import *
from DecisionTreeCART import *
import time
import numpy as np
import random
DataSet = [['youth', 'no', 'no', '1', 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', '1', 'agree'],
               ['youth', 'no', 'no', '1', 'refuse'],
               ['mid', 'no', 'no', '1', 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', '1', 'refuse'],
               ]
Labels = ['age', 'working?', 'house?', 'credit_situation']

TEST_DATA = [['young', 'hyper', 'yes', 'reduced', 'no lenses'],
             ['young', 'hyper', 'yes', 'normal', 'no lenses'],
             ['pre', 'myope', 'yes', 'normal', 'hard'],
             ['presbyopic', 'hyper', 'no', 'normal', 'soft'],
             ['presbyopic', 'hyper', 'no', 'normal', 'soft'],
             ]


def load_data(file_path="test.csv", encoding="utf-8"):
    data_set = []
    with open(file_path, encoding=encoding) as fp:
        csv_reader = csv.reader(fp)
        for ri, row in enumerate(csv_reader):
            if ri == 0:
                labels = row
            else:
                data_set.append(row)
    data_set = np.array(data_set)
    entry_nums = len(data_set)
    train_nums = int(entry_nums * 0.8)
    test_nums = entry_nums - train_nums
    train_index = random.sample(list(range(entry_nums)), train_nums)
    test_index = list(set(range(entry_nums)) - set(train_index))
    train_data = data_set[train_index].tolist()
    test_data = data_set[test_index].tolist()
    return labels, train_data, test_data, data_set.tolist()


if __name__ == "__main__":
    labels, train_data, test_data, data_set = load_data("car_data.txt")
    decision_tree = DecisionTreeC45(data_set=train_data, labels=labels)  # types=["str", "float", "float", "str",])
    # decision_tree.load_data("car_data.txt")
    test_time = 100
    time_add = 0.0
    while test_time:
        test_time -= 1
        bg = time.time()
        decision_tree.set_up()
        end = time.time()
        time_add += (end - bg)
    print("set up time:\t", time_add/100)
    decision_tree.pruning(test_data)
    print(decision_tree)
    print("depth:\t", decision_tree.get_depth())
    print("leaf numbers:\t", decision_tree.get_leaf_nums())
    print("error count:\t", decision_tree.calc_err(test_data))
    print("accuracy:\t", decision_tree.calc_accuracy(test_data))
    decision_tree.view_in_graph()
