import csv
import node_plot as nplt
from class_compute import *
import re


def get_leaf_ext(tree):
    if type(tree).__name__ != "dict":
        return 1
    if tree == {}:
        return 0
    tree_queue = [tree]
    leaf_nums = 0
    while len(tree_queue):
        queue_len = len(tree_queue)
        while queue_len:
            queue_len = queue_len - 1
            p_tree = tree_queue.pop(0)
            node_label = list(p_tree.keys())[0]
            for key in p_tree[node_label].keys():
                if type(p_tree[node_label][key]).__name__ == "dict":
                    tree_queue.append(p_tree[node_label][key])
                else:
                    leaf_nums += 1
    return leaf_nums


def match_binary_key(keys, key_val):
    if keys[0].find("<=") != -1 or keys[0].find(">") != -1:  # 非字串分类
        try:
            thr = float(keys[0].replace("<=", ""))
        except:
            thr = float(keys[0].replace(">", ""))
        if float(key_val) <= thr:
            return keys[0]
        else:
            return keys[1]
    else:
        for key in keys:
            key_split = key.split('\n')
            if key_val in key_split:
                return key


def match_decision_key(keys, key_val):
    """match feat val  in correct form in tree dict, eg. float value 30 in cart tree key form is  <=30 or > 30
    #parm:
    keys: tree dict key
    key_val: feat val
    return: dict key"""
    if keys[0].find("<=") != -1 or keys[0].find(">") != -1:  # 非字串分类
        thr = re.match("\W*(\w+)", key_val).group(1)
        if float(key_val) <= thr:
            return "<=%s" % key_val
        else:
            return ">%s" % key_val
    else:
        for key in keys:
            key_split = key.split('\n')
            if key_val in key_split:
                return key


def match_key_val(key):
    """match get key feature from tree dict, eg. >30 to 30
    #param:
    key: tree dict key
    return : feature value"""
    return re.match("\W*(\w+)", key).group(1)


def classify(tree, labels, test_data, binary=False):
    if tree == {}:
        raise Exception("Decision tree doesn't exist")
    while type(tree).__name__ == "dict":
        node_label = list(tree.keys())[0]
        val = test_data[labels.index(node_label)]
        if binary:
            tree = tree[node_label][match_decision_key(list(tree[node_label].keys()), val)]
            continue
        tree = tree[node_label][val]
    return tree


def calc_err_ext(tree, labels, test_data, binary):
    err_cnt = 0.0
    for i in range(len(test_data)):
        try:
            if classify(tree, labels, test_data[i], binary) != test_data[i][-1]:
                err_cnt += 1
        except:
            continue

    return err_cnt


class DecisionTree(object):

    def __init__(self, data_set=list(), labels=list(), file_path=None, ops=(1, 0.001), encoding="utf-8"):
        self.data_set = data_set
        self.labels = labels
        self.tree = dict()
        # self._vec_nums = None
        self._attr_nums = None
        self.binary = False
        self.err_cnt = 0.0
        self.err_rate = 0.0 
        self.ops = ops
        if file_path is not None:
            self.load_data(file_path, encoding)
        self.data_nums = len(self.data_set)
        self.class_list = list(set([vec[-1] for vec in self.data_set]))

    def load_data(self, file_path="test.csv", encoding="utf-8"):
        with open(file_path, encoding=encoding) as fp:
            csv_reader = csv.reader(fp)
            for ri, row in enumerate(csv_reader):
                if ri == 0:
                    self.labels = row
                else:
                    self.data_set.append(row)
        self.data_nums = len(self.data_set)
        self.class_list = list(set([vec[-1] for vec in self.data_set]))

    def set_up(self):
        pass

    def get_depth(self):
        if self.tree == {}:
            return 0
        tree_queue = [self.tree]
        depth = 0
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                node_label = list(p_tree.keys())[0]
                for key in p_tree[node_label].keys():
                    if type(p_tree[node_label][key]).__name__ == "dict":
                        tree_queue.append(p_tree[node_label][key])
            depth += 1

        return depth + 1   # leaf node will not attend in queue , so the depth must plus 1

    def get_leaf_nums(self):
        if self.tree == {}:
            return 0
        tree_queue = [self.tree]
        leaf_nums = 0
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                node_label = list(p_tree.keys())[0]
                for key in p_tree[node_label].keys():
                    if type(p_tree[node_label][key]).__name__ == "dict":
                        tree_queue.append(p_tree[node_label][key])
                    else:
                        leaf_nums += 1
        return leaf_nums

    def view_in_graph(self):
        if self.tree == {}:
            print("nothing to plot, back !")
            return
        height = self.get_depth()
        width = self.get_leaf_nums()
        tree_wh = {"height": height, "width": width}

        tree_queue = [self.tree]
        depth_level = 0

        ax = nplt.draw_init(1)
        class_num = len(self.class_list)

        for ix in range(class_num):
            coord_lgd = (ix/(2*class_num), 1)
            nplt.plot_node(ax, self.class_list[ix], coord_lgd, coord_lgd, ix + 1)  # leaf node legend

        parent_point = (0.5, 1 - 1.0/(2 * tree_wh["height"]))
        point_queue = [parent_point]
        arrow_txt_queue = ['']
        while len(tree_queue):
            queue_len = len(tree_queue)
            x_offset = -1.0 / (2 * tree_wh["width"])
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                parent_point = point_queue.pop(0)
                arrow_txt = arrow_txt_queue.pop(0)
                leafn = get_leaf_ext(p_tree)
                center_point = nplt.get_coord(x_offset, depth_level, leafn, tree_wh)
                # print(center_point)
                if leafn == 0:  # 当前层的空点，其上层父节点是叶子节点， 将x_offset 偏移一个节点：
                    x_offset += 1 / tree_wh["width"]
                    if depth_level < tree_wh["height"]:
                        arrow_txt_queue.append('')
                        point_queue.append(center_point)
                        tree_queue.append({})
                    continue
                if leafn == 1:  # 当前层的叶节点，画叶节点， 将x_offset 偏移一个节点
                    node_style = self.class_list.index(str(p_tree)) + 1
                    # nplt.plot_node(ax, str(p_tree), center_point, parent_point, node_style)
                    nplt.plot_node(ax, str(p_tree)[0], center_point, parent_point, node_style)
                    nplt.plot_text(ax, center_point, parent_point, arrow_txt)
                    x_offset += 1 / tree_wh["width"]
                    if depth_level < tree_wh["height"] - 1:
                        arrow_txt_queue.append('')
                        point_queue.append(center_point)
                        tree_queue.append({})
                    continue

                # 当前层的子树， 继续子节点/子树入队， 并画节点
                node_label = list(p_tree.keys())[0]
                nplt.plot_node(ax, node_label, center_point, parent_point, 0)
                nplt.plot_text(ax, center_point, parent_point, arrow_txt)
                x_offset += leafn / tree_wh["width"]

                arrow_txt_queue2 = []
                point_queue2 = []
                tree_queue2 = []
                for key in p_tree[node_label].keys():  # 让子树结构先进队再进子结点
                    if type(p_tree[node_label][key]).__name__ == "dict":
                        arrow_txt_queue.append(key)
                        point_queue.append(center_point)
                        tree_queue.append(p_tree[node_label][key])
                    else:
                        arrow_txt_queue2.append(key)
                        point_queue2.append(center_point)
                        tree_queue2.append(p_tree[node_label][key])

                arrow_txt_queue.extend(arrow_txt_queue2)
                point_queue.extend(point_queue2)
                tree_queue.extend(tree_queue2)
            depth_level += 1
        nplt.show()

    def calc_err(self, test_data=None):
        if test_data is None:
            test_data = self.data_set
        self.err_cnt = calc_err_ext(self.tree, self.labels, test_data, self.binary)
        self.err_rate = self.err_cnt/(len(test_data))
        return self.err_cnt

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
        while len(tree_queue):
            queue_len = len(tree_queue)
            while queue_len:
                queue_len = queue_len - 1
                p_tree = tree_queue.pop(0)
                data_set = data_queue.pop(0)
                labels = label_queue.pop(0)
                node_label = list(p_tree.keys())[0]
                feat = labels.index(node_label)
                del(labels[feat])
                sub_labels = labels
                res_data = split_data_entropy(data_set, feat)
                # for key in p_tree[node_label].keys():
                for key in res_data.keys():  # 只care 存在测试数据的分支
                    # if len(res_data[key]) == 0:  # empty split
                    #     continue
                    if key not in p_tree[node_label].keys():
                        print("%s get an unexpected value %s " % (node_label, key))
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
        i = 0
        while len(tree_stack):
            i = i + 1
            if i == 2:
                pass
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

    def calc_accuracy(self, data_set):
        entry_nums= len(data_set)
        error = calc_err_ext(self.tree, self.labels, data_set, self.binary)
        accuracy = 1.0 - float(error)/entry_nums
        return accuracy*100.0


    def __str__(self):
        return str(self.tree)
