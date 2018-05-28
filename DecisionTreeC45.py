__author__ = "huangshiyang"


from class_compute import *
from DecisionTree import *


class DecisionTreeC45(DecisionTree):
    def __init__(self, data_set=list(), labels=list(), file_path=None, encoding="utf-8"):
        super(DecisionTreeC45, self).__init__(data_set, labels, file_path, encoding)
        self.method = "C4.5"

    def set_up(self):
        tree_queue = [self.tree]
        data_queue = [self.data_set]
        labels_queue = [self.labels.copy()]
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                data_set = data_queue.pop(0)
                labels = labels_queue.pop(0)
                best_feat = choose_best_feat_entropy(data_set, self.method)
                best_feat_label = labels[best_feat]
                p_tree[best_feat_label] = {}
                del(labels[best_feat])
                sub_labels = labels
                res_data = split_data_entropy(data_set, best_feat)
                for son_label in res_data.keys():
                    son_val = get_class_value(res_data[son_label])
                    p_tree[best_feat_label][son_label] = son_val
                    if type(son_val).__name__ == "dict":
                        tree_queue.append(p_tree[best_feat_label][son_label])
                        data_queue.append(res_data[son_label])
                        labels_queue.append(sub_labels.copy())



















