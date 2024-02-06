# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle
import csv
from typing import List

agree_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
neutral_dict = {'case': 0, 'wh' : 0, 'foc' : 0}

# Custom node class with a named list field
class SyntaxNode(Node):
    def __init__(self, name, label = None, merge_feat = None, neutral_feats = None, agree_feats = None, domination_count = None, parent = None, children = None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else {}
        self.merge_feat = merge_feat if merge_feat is not None and merge_feat != '' else {}
        self.domination_count = domination_count if domination_count is not None else 0
        self.agree_feats = agree_feats if agree_feats is not None else agree_dict
        self.neutral_feats = neutral_feats if neutral_feats is not None else neutral_dict

# clone a node to avoid myltiple assignment
def clone_tree(node):
    cloned_node = SyntaxNode(
        name=node.name,
        label=node.label,
        merge_feat = node.merge_feat,
        agree_feats=node.agree_feats,
        neutral_feats=node.neutral_feats
        # Add other attributes as needed
    )
    for child in node.children:
        cloned_child = clone_tree(child)
        cloned_child.parent = cloned_node
    return cloned_node    

def parse_feats(feat_str):
    if feat_str is not None and feat_str.strip():
        # Split the string by '-' and then create a dictionary from key-value pairs
        return dict(pair.split(':') for pair in feat_str.split('-'))
    else:
        return {}

# read nodes from a csv file
def read_nodes_csv(csv_file_path: str) -> List[SyntaxNode]: 
    nodes = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node = SyntaxNode(
                name=row['it'],
                merge_feat=(row['mc']),
                agree_feats=parse_feats(row['ac']),
                neutral_feats=parse_feats(row['ft']),
                label=(row['lb'])
            )
            nodes.append(node)
    return nodes

#### GEN FUNCTIONS ####
# Merge function
def Merge(right_arg, node_list):
    output_nodes = []
    for node in node_list:
        new_left = clone_tree(node)
        new_right = clone_tree(right_arg)
        new_node = SyntaxNode("")
        new_node.children = [new_left, new_right]
        output_nodes.append(new_node)
    return output_nodes

# Label function
def Label(my_node):
    my_nodes = []
    if not my_node.name and len(my_node.children) == 2:
        # take from left
        new_1 = clone_tree(my_node)
        new_1.merge_feat = {}
        new_1.name = new_1.children[0].name
        new_1.label = new_1.children[0].label
        new_1.agree_feats = new_1.children[0].agree_feats
        new_1.neutral_feats = new_1.children[0].neutral_feats
        my_nodes.append(new_1)

        # take from right
        new_2 = clone_tree(my_node)
        new_2.merge_feat = {}
        new_2.name = new_2.children[1].name
        new_2.label = new_2.children[1].label
        new_2.agree_feats = new_2.children[1].agree_feats
        new_2.neutral_feats = new_2.children[1].neutral_feats
        my_nodes.append(new_2)
    return(my_nodes)

# Agree function, only under sisterhood
def Agree(my_node):
    if not my_node.name and len(my_node.children) > 1: # if the root node is not labeled and has more than one child
        new_node = clone_tree(my_node)
        # agree left
        my_left_agr = my_node.children[0].agree_feats
        my_right_feats = my_node.children[1].neutral_feats
        for key in my_right_feats:
            if key in my_left_agr and my_right_feats[key] == my_left_agr[key]:
                my_right_feats[key] = my_left_agr[key] = 0
        new_node.children[0].agree_feats = my_left_agr     
        new_node.children[1].neutral_feats = my_right_feats

        # agree right
        my_right_agr =  my_node.children[1].agree_feats
        my_left_feats = my_node.children[0].neutral_feats
        for key in my_left_feats:
            if key in my_right_agr and my_left_feats[key] == my_right_agr[key]:
                my_left_feats[key] = my_right_agr[key] = 0
        new_node.children[1].agree_feats = my_right_agr     
        new_node.children[0].neutral_feats = my_left_feats

        # return a list
        return([new_node])

# import numeration
my_nodes = read_nodes_csv("./unaccusative_numeration.csv")

# carry out first Merge
my_result = Merge(my_nodes[0], my_nodes[1:])

trial = Agree(my_result[2])

# Visualize the tree using ASCII art
for pre, _, node in RenderTree(trial[0], style=AsciiStyle()):
    print(f"{pre}{node.name} - {node.label} - {node.merge_feat} - {node.agree_feats} - {node.neutral_feats}")
