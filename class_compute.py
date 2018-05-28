from math import log2
import operator
import numpy as np

def xlog(prob):
    return prob * log2(prob) if prob else 0.0


def test_major(major, test_data):
    error_count = 0.0
    for i in range(len(test_data)):
        if major != test_data[i][-1]:
            error_count += 1
    return error_count


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:  # 统计所有类标签的频数
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    max_count_label = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)[0][0]  # 排序
    return max_count_label


def get_class_value(data_set):
    class_list = [entry[-1] for entry in data_set]
    # try:
    if class_list.count(class_list[0]) == len(class_list):  # no need classify
        return class_list[0]
    elif len(data_set[0]) == 1:  # only class col exists , finished
        return majority_cnt(class_list)
    else:  # be a sub tree
        return dict()


def calc_entropy(data_set):
    labels_count = {}
    # 统计每个类别的数量
    entry_nums = len(data_set)
    for feat_vec in data_set:
        class_label = feat_vec[-1]
        if class_label not in labels_count.keys():
            labels_count[class_label] = 0
        # print(labels_count[class_label])
        # print(type(labels_count[class_label]).__name__)
        labels_count[class_label] += 1

    entropy = 0.0
    for label in labels_count:
        prob = float(labels_count[label]) / entry_nums
        entropy -= xlog(prob)
    return entropy


def calc_gini(data_set):
    labels_count = {}
    # 统计每个类别的数量
    entry_nums = len(data_set)
    for feat_vec in data_set:
        class_label = feat_vec[-1]
        if class_label not in labels_count.keys():
            labels_count[class_label] = 0
        # print(labels_count[class_label])
        # print(type(labels_count[class_label]).__name__)
        labels_count[class_label] += 1

    gini = 1.0
    for label in labels_count:
        prob = float(labels_count[label]) / entry_nums
        gini -= prob*prob
    return gini


def split_data_entropy(data_set, col):
    res_set = {}
    for feat_vec in data_set:
        # tmpLine = featVec[:col]
        # tmpLine.extend(featVec[col + 1:])
        res_vec = feat_vec[:col]
        res_vec.extend(feat_vec[col+1:])

        col_val = feat_vec[col]
        if col_val not in res_set.keys():
            res_set[col_val] = [res_vec]
        else:
            res_set[col_val].append(res_vec)
    return res_set


def pack_to_label(feat_redu):
    if feat_redu == set():
        return ''
    label = list(feat_redu)[0]
    for feat in list(feat_redu)[1:]:
        label = label + '\n' + str(feat)
    if len(label) > 20:
        return label[0:20] + '...'
    return label


def split_data_val(data_set, col, value):
    res_data = []
    for feat_vec in data_set:
        if feat_vec[col] == value:
            res_vec = feat_vec[:col]
            res_vec.extend(feat_vec[col + 1:])
            res_data.append(res_vec)
    return res_data


def split_data_cart(data_set, col, value, types, binary_split=False):
    if types == "str":
        return split_str_gini(data_set, col, value, binary_split)
    else:
        return split_float_gini(data_set, col, value)


# def match_data_split(tree, data_set, col, value, types, binary_split):
#     if types == "str":
#         return split_float_gini(data_set, col, value, binary_split)
#     else:
#         return split_float_gini(data_set, col, value)


def split_float_gini(data_set, col, value):
    d0 = []
    d1 = []
    if len(data_set) == 0:
        return None
    for feat_vec in data_set:
        res_vec = feat_vec[:col]
        res_vec.extend(feat_vec[col + 1:])
        if float(feat_vec[col]) <= float(value):
            d0.append(res_vec)
        else:
            d1.append(res_vec)
    label0 = "<=" + str(value)
    label1 = ">" + str(value)
    res_data = {label0: d0, label1: d1}
    return res_data


def split_str_gini(data_set, col, value, binary_split):
    d0 = []
    d1 = []
    if len(data_set) == 0:
        return None
    if not binary_split:  # 按属性值划分
        return split_data_entropy(data_set, col)
    # 二分类
    feat_vals = set([vec[col] for vec in data_set])
    feat_redu = feat_vals - {value}
    feat_redu_label = pack_to_label(feat_redu)
    for feat_vec in data_set:
        res_vec = feat_vec[:col]
        res_vec.extend(feat_vec[col + 1:])
        if feat_vec[col] == value:
            d0.append(res_vec)
        else:
            d1.append(res_vec)
    res_data = {str(value): d0, feat_redu_label: d1}
    return res_data


def calc_gini_with_feat(data_set, feature, value, types):
    d0 = []
    d1 = []
    if types != "str":
        for feat_vec in data_set:
            if float(feat_vec[feature]) <= float(value):
                d0.append(feat_vec)
            else:
                d1.append(feat_vec)
    else:
        for feat_vec in data_set:
            if feat_vec[feature] == value:
                d0.append(feat_vec)
            else:
                d1.append(feat_vec)
    gini = len(d0) / len(data_set) * calc_gini(d0) + len(d1) / len(data_set) * calc_gini(d1)
    return gini


def calc_conditional_entropy(data_set, i):
    condition_ent = 0.0
    entry_nums = len(data_set)
    res_data = split_data_entropy(data_set, i)

    for key in res_data.keys():
        sub_data = res_data[key]
        prob = float(len(sub_data))/entry_nums
        condition_ent += prob * calc_entropy(sub_data)

    return condition_ent


def calc_info_gain(base_ent, data_set, i):
    condition_ent = calc_conditional_entropy(data_set, i)
    info_gain = base_ent - condition_ent
    return info_gain


def calc_info_gain_ratio(base_ent, data_set,  i):
    return calc_info_gain(base_ent, data_set, i)/base_ent


def choose_best_feat_entropy(data_set, method):
    if method == "ID3":
        calc_func = calc_info_gain
    elif method == "C4.5":
        calc_func = calc_info_gain_ratio
    else:
        return
    best_info_gain = 0.0
    best_feat = -1
    base_ent = calc_entropy(data_set)
    feat_nums = len(data_set[0]) - 1
    for i in range(feat_nums):
        info_gain = calc_func(base_ent, data_set, i)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


def choose_best_feat_gini(data_set, types=None):
    feat_nums = len(data_set[0]) - 1
    best_gini = np.inf
    best_feat = 0
    best_split_val = None
    if types is None:
        types = ["str"] * feat_nums
    for i in range(feat_nums):
        feat_list = [vec[i] for vec in data_set]
        feat_set = set(feat_list)
        for split_val in feat_set:
            new_gini = calc_gini_with_feat(data_set, i, split_val, types[i])
            if new_gini < best_gini:
                best_feat = i
                best_gini = new_gini
                best_split_val = split_val
    return best_feat, best_split_val

