import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree

''''' 数据读入 '''
data = []
labels = []
feat_list = []
with open("car_data.txt") as ifile:
    for il, line in enumerate(ifile):
        if il == 0:
            feat_list = line.strip().split(',')
            continue
        tokens = line.strip().split(',')
        data.append(tokens[:-1])
        labels.append(tokens[-1])


x = np.array(data)
labels = np.array(labels)
y = labels.reshape((x.shape[0], 1))

''''' 标签转换为0/1 '''
# y[labels == 'hard'] = 1
# y[labels == 'soft'] = -1

# feat_nums = x.shape[1]
# feat_map = []
# for i in range(feat_nums):
#     value_set = set(x[:, i].tolist())
#     j = 0
#     feat_dict = {}
#     for value in value_set:
#         feat_dict[j] = value
#         x[x[:, i] == value, i] = j
#         j = j + 1
#     feat_map.append(feat_dict)


def map_str_label(data_mat):
    feat_nums = data_mat.shape[1]
    feat_map = []
    for i in range(feat_nums):
        value_set = set(data_mat[:, i].tolist())
        j = 0
        feat_dict = {}
        for value in value_set:
            feat_dict[j] = value
            data_mat[data_mat[:, i] == value, i] = j
            j = j + 1
        feat_map.append(feat_dict)
    return data_mat, feat_map
''''' 拆分训练数据与测试数据 '''
x, x_label = map_str_label(x)
y, y_label = map_str_label(y)
y = y.reshape(y.shape[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

class_names = []
for class_label in y_label[0].values():
    class_names.append(class_label)
''''' 把决策树结构写入文件 '''
with open("car_data.dot", 'w+') as f:
    f = tree.export_graphviz(clf, feature_names=feat_list, class_names=class_names, rounded=True, out_file=f)

''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''''测试结果的打印'''
answer = clf.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))


