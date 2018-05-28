__author__ = "huangshiyang"


from class_compute import *
from DecisionTree import *
import re

class DecisionTreeCART(DecisionTree):
    def __init__(self, data_set=list(), labels=list(), binary=False, types=None, file_path=None, encoding="utf-8"):
        super(DecisionTreeCART, self).__init__(data_set, labels, file_path, encoding)
        self.method = "CART"
        self.binary = binary
        self.types = types

    # def set_up_nobinary(self):
    #     if self.types is None:
    #         self.types = ['str'] * (len(self.data_set[0]) - 1)
    #     tree_queue = [self.tree]
    #     data_queue = [self.data_set]
    #     labels_queue = [self.labels.copy()]
    #     while len(tree_queue):
    #         queue_len = len(tree_queue)
    #         while queue_len:
    #             queue_len = queue_len - 1
    #             p_tree = tree_queue.pop(0)
    #             data_set = data_queue.pop(0)
    #             labels = labels_queue.pop(0)
    #             best_feat, _ = choose_best_feat_gini(data_set)
    #             best_feat_label = labels[best_feat]
    #             p_tree[best_feat_label] = {}
    #             del (labels[best_feat])
    #             sub_labels = labels
    #             res_data = split_data_cart(data_set, best_feat)
    #             for son_label in res_data.keys():
    #                 son_val = get_class_value(res_data[son_label])
    #                 p_tree[best_feat_label][son_label] = son_val
    #                 if type(son_val).__name__ == "dict":
    #                     tree_queue.append(p_tree[best_feat_label][son_label])
    #                     data_queue.append(res_data[son_label])
    #                     labels_queue.append(sub_labels.copy())

    # def set_up_binary(self):
    def set_up(self):
        if self.types is None:
            self.types = ['str'] * (len(self.data_set[0]) - 1)
        tree_queue = [self.tree]
        data_queue = [self.data_set]
        labels_queue = [self.labels.copy()]
        types_queue = [self.types.copy()]  # 列数据属性，如果是连续型数字采用回归分法
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                data_set = data_queue.pop(0)
                labels = labels_queue.pop(0)
                types = types_queue.pop(0)
                best_feat, best_split_val = choose_best_feat_gini(data_set, types)
                best_feat_label = labels[best_feat]
                p_tree[best_feat_label] = {}
                del (labels[best_feat])
                sub_labels = labels
                res_data = split_data_cart(data_set, best_feat, best_split_val, types[best_feat], self.binary)
                del (types[best_feat])
                sub_types = types
                for son_label in res_data.keys():
                    son_val = get_class_value(res_data[son_label])
                    p_tree[best_feat_label][son_label] = son_val
                    if type(son_val).__name__ == "dict":
                        tree_queue.append(p_tree[best_feat_label][son_label])
                        data_queue.append(res_data[son_label])
                        labels_queue.append(sub_labels.copy())
                        types_queue.append(sub_types.copy())

    # def set_up(self):
    #     self.set_up_binary()
        # if self.binary is True:
        #     self.set_up_binary()
        # else:
        #     self.set_up_nobinary()

    def pruning(self, test_data=None):
        if self.tree == {}:
            return
        if test_data is None:
            test_data = self.data_set
        tree_queue = [self.tree]
        tree_stack = []
        data_queue = [test_data]
        data_stack = []
        label_queue = [self.labels.copy()]
        label_stack = []
        key_stack = []
        types_queue = [self.types.copy()]
        # class_list = [vec[-1] for vec in test_data]
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                data_set = data_queue.pop(0)
                labels = label_queue.pop(0)
                types = types_queue.pop(0)
                node_label = list(p_tree.keys())[0]
                feat = labels.index(node_label)
                del(labels[feat])
                sub_labels = labels

                key_val = list(p_tree[node_label].keys())[0]
                if types[feat] != "str":
                    key_val = match_key_val(key_val)

                res_data = split_data_cart(data_set, feat, key_val, types[feat], self.binary)

                del (types[feat])
                sub_types = types
                # for key in p_tree[node_label].keys():
                for key in res_data.keys():
                    if len(res_data[key]) == 0:  # empty split
                        continue
                    if key not in p_tree[node_label].keys():
                        print("%s get an unexpected value %s " %(node_label, key))
                        continue
                    if type(p_tree[node_label][key]).__name__ == "dict":
                        sub_data = res_data[key]
                        tree_queue.append(p_tree[node_label][key])
                        key_stack.append(key)
                        tree_stack.append(p_tree[node_label])
                        data_queue.append(sub_data)
                        data_stack.append(sub_data)
                        label_queue.append(sub_labels.copy())
                        label_stack.append(sub_labels.copy())
                        types_queue.append(sub_types.copy())
        print("len stack", len(tree_stack))
        # i = 0
        while len(tree_stack):
            # print(i)
            # i = i + 1
            key = key_stack.pop()
            p_tree = tree_stack.pop()
            data_set = data_stack.pop()
            labels = label_stack.pop()
            if len(data_set):
                class_list = [vec[-1] for vec in data_set]
                err1 = calc_err_ext(p_tree[key], labels, data_set, self.binary)
                err2 = test_major(majority_cnt(class_list), data_set)
                print("err1:\t", err1)
                print("err2:\t", err2)
                if err1 < err2:
                    continue
                else:
                    p_tree[key] = majority_cnt(class_list)
            else:
                continue
        return

















