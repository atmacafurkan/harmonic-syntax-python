# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, PostOrderIter
import csv
from typing import List

agree_dict = {'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0}
neutral_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
constraints_dict = {'label_cons': 0, 'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0, 'case': 0, 'wh' : 0, 'foc' : 0}

# Custom node class with a named list field
class SyntaxNode(Node):
    def __init__(self, name, label = None, merge_feat = None, neutral_feats = None, agree_feats = None, parent = None, children = None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else None
        self.merge_feat = merge_feat if merge_feat is not None and merge_feat != '' else None
        self.agree_feats = agree_feats if agree_feats is not None else agree_dict
        self.neutral_feats = neutral_feats if neutral_feats is not None else neutral_dict
        self.other_nodes = []

    def add_other_node(self, other_node):
        self.other_nodes.append(other_node)
            
    def domination_count(self):
        # Get all distinct parent labels recursively
        parent_labels = set()
        current_parent = self.parent

        while current_parent:
            if current_parent.label and current_parent.label != self.label:
                parent_labels.add(current_parent.label)
            current_parent = current_parent.parent
        # Return the count of distinct parent labels
        return len(parent_labels)

    def evaluate_constraints(self):
        # Initialize a result dictionary for both agree_feats and neutral_feats
        result_feats = constraints_dict.copy()

        # Check if any ancestor has the same name
        current_ancestor = self.parent
        while current_ancestor:
            if current_ancestor.name == self.name:
                return result_feats
            current_ancestor = current_ancestor.parent
        
        # add labelling constraint
        if not self.name:
            result_feats['label_cons'] = 0

        # Multiply agree_feats and neutral_feats with domination_count for the root node
        root_domination_count = self.domination_count()

        # Update result_feats with the values for the root node
        for key, value in self.agree_feats.items():
            result_feats[key] = int(value) * root_domination_count

        for key, value in self.neutral_feats.items():
            result_feats[key] = int(value) * root_domination_count

        return result_feats
    
    def merge_condition(self):
        merge_condition_results = {'merge_cond': 0}
        if len(self.children) > 1:
            if self.children[0].merge_feat and self.children[0].merge_feat != self.children[1].label:
                merge_condition_results['merge_cond'] += 1

            if self.children[1].merge_feat and self.children[1].merge_feat != self.children[0].label:
                merge_condition_results['merge_cond'] += 1 

        return merge_condition_results
    
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

def traverse_and_clone(node, cloned_nodes):
    # Base case: If the node is None or has already been cloned, return
    if node is None or node in cloned_nodes:
        return

    # Clone the current node
    cloned_node = clone_tree(node)
    # Add the cloned node to the list of cloned nodes
    cloned_nodes.add(cloned_node)

    # Recursively traverse and clone the children of the current node
    for child in node.children or []:
        traverse_and_clone(child, cloned_nodes)   

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
                label=row['lb']
            )
            nodes.append(node)

    # Update the other_nodes field for each node by cloning the other nodes
    for i, node in enumerate(nodes):
        node.other_nodes = [clone_tree(other_node) for j, other_node in enumerate(nodes) if i != j]

    return nodes

#### GEN FUNCTIONS ####
# Merge function
def Merge(my_arg):
    output_nodes = []
    node_list = my_arg.other_nodes

    # Set to keep track of cloned nodes to avoid duplicates
    cloned_nodes = set()

    # Traverse and clone every distinct node of my_arg
    if len(my_arg.children) > 1:
        traverse_and_clone(my_arg, cloned_nodes)
    else:
        cloned_nodes = []
    # The loops are separated for convenience, what is done is the same.
    # external merge
    for original_node in node_list:
        new_right = clone_tree(my_arg)
        new_node = SyntaxNode("")

        # Omit the node that is being used from node_list
        new_node.other_nodes = [clone_tree(n) for n in node_list if n != original_node]

        # Remove the other_nodes of the children
        new_right.other_nodes = []

        # Join the original_node and new_right under the new node
        new_node.children = [clone_tree(original_node), new_right]
        
        # Set label if conditions are met
        if original_node.label == new_right.label:
            new_node.label = original_node.label

        output_nodes.append(new_node)

    # Separate loop over cloned nodes
    # internal merge
    for cloned_node in cloned_nodes:
        new_left = clone_tree(cloned_node)
        new_right = clone_tree(my_arg)
        new_node = SyntaxNode("")

        # copy the other_nodes of input entirely, all merges are internal
        new_node.other_nodes = [clone_tree(n) for n in my_arg.other_nodes]

        # Remove the other_nodes of the children
        new_right.other_nodes = []

        # Join the cloned_node and new_right under the new node
        new_node.children = [new_left, new_right]

        # Set label if conditions are met
        if new_left.label == new_right.label:
            new_node.label = new_left.label

        output_nodes.append(new_node)

    return output_nodes

# Label function
def Label(my_node):
    my_nodes = []
    if not my_node.name and len(my_node.children) == 2:
        # take from left
        new_1 = clone_tree(my_node)
        new_1.other_nodes = my_node.other_nodes
        new_1.merge_feat = {}
        new_1.name = new_1.children[0].name
        new_1.label = new_1.children[0].label
        new_1.agree_feats = new_1.children[0].agree_feats
        new_1.neutral_feats = new_1.children[0].neutral_feats
        my_nodes.append(new_1)

        # take from right
        new_2 = clone_tree(my_node)
        new_2.other_nodes = my_node.other_nodes
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
    else: # if the root node is labeled or has less than two children
        new_node = []

# import numeration
my_nodes = read_nodes_csv("./unaccusative_numeration.csv")
my_result = Merge(Label(Merge(my_nodes[0])[0])[2])[4]

# Visualize the tree using ASCII art
for pre, _, node in RenderTree(my_result, style=AsciiStyle()):
    print(f"{pre}{node.name} - {node.agree_feats} - {node.neutral_feats} - {node.merge_condition()} - {node.evaluate_constraints()} - {node.merge_feat} - {node.label}")


