"""this module define some drawing styles and interface of BiNode"""
import matplotlib.pyplot as plt

DECISION_NODE = dict(boxstyle="round4,pad=0.3", fc="cyan", ec="black", lw=1)
LEAF_NODE = dict(boxstyle="circle", color='orange', alpha=0.5)  # 定义叶结点形态
ARROW_ARGS = dict(arrowstyle="<-", color='g')
COLORS_SET = ("orange", "yellow", "green",  "blue", "purple", "pink", "azure", "beige", "coral")


def leaf_node(x):
    LEAF_NODE["color"] = COLORS_SET[x]
    return LEAF_NODE


def draw_init(fig_num=1):
    f = plt.figure(fig_num)
    ax = f.add_subplot(111)
    return ax

#
# def get_coord(coord_prt, depth_le, depth, child_type="left"):
#     if child_type == "left":
#         x_child = coord_prt[0] - 1 / (2 ** (depth_le + 1))
#     elif child_type == "right":
#         x_child = coord_prt[0] + 1 / (2 ** (depth_le + 1))
#     else:
#         raise Exception("No other child type")
#     y_child = coord_prt[1] - 1 / depth
#     return x_child, y_child


def get_coord(x_offset, depth_le, leaf_nums, tree_wh):
    tree_width = tree_wh["width"]
    tree_height = tree_wh["height"]
    d = (1 - 1/(2 * tree_height))*20/((tree_height-1)*(tree_height + 18))
    y_offset = (1.1 * depth_le - 0.1) if depth_le > 0 else 0
    ycoord = 1 - 1/(2 * tree_height) - y_offset * d
    # ycoord = 1 - (2*depth_le + 1)/(2*tree_height)
    xcoord = x_offset + (leaf_nums + 1)/(2*tree_width)
    return xcoord, ycoord


def plot_text(ax, center_point, parent_point, txt_string):
    x_mid = (parent_point[0]-center_point[0])/2.0 + center_point[0]
    y_mid = (parent_point[1]-center_point[1])/2.0 + center_point[1]
    ax.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=45)


def plot_node(ax, node_text, center_point, parent_point, node_style):
    if node_style == 0:
        ax.annotate(node_text, xy=parent_point,  xycoords='axes fraction', xytext=center_point, fontsize=10,
                    textcoords='axes fraction', va="bottom", ha="center", bbox=DECISION_NODE, arrowprops=ARROW_ARGS)
    else:
        ax.annotate(node_text, xy=parent_point, xycoords='axes fraction', xytext=center_point, fontsize=10,
                    textcoords='axes fraction', va="bottom", ha="center", bbox=leaf_node(node_style - 1),
                    arrowprops=ARROW_ARGS)


def show():
    plt.title("Graph of DecisionTree")
    plt.axis("off")
    plt.show()

