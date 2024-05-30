# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle
import csv
from typing import List
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from anytree.exporter import DotExporter
import os
import sys

agree_dict = {'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0}
neutral_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
used_feats_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
constraints_dict = {'label_cons': 1, 'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0, 'case': 0, 'wh' : 0, 'foc' : 0}

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

    def evaluate_constraints(self, encountered_nodes=None, default_state = None):
        # Initialize a result dictionary for both agree_feats and neutral_feats
        result_feats = constraints_dict.copy() if default_state is None else default_state

        # Initialize encountered_nodes if not provided
        encountered_nodes = set() if encountered_nodes is None else encountered_nodes

        # Add the current node's name to the set of encountered nodes
        encountered_nodes.add(self.name)

        # Multiply agree_feats and neutral_feats with domination_count for the current node
        domination_count = self.domination_count()

        # agree feature violations
        for key, value in self.agree_feats.items():
            result_feats[key] = int(value) * domination_count

        # neutral feature violations
        for key, value in self.neutral_feats.items():
            result_feats[key] = int(value) * domination_count

        # If encountered before, reset evaluation
        if self.name in encountered_nodes:
            result_feats = constraints_dict.copy()

        # Add labelling constraint # TURN INTO A SEPARATE FUNCTION!!!
        if self.name:
            result_feats['label_cons'] = 0
        
        # Recursive call for each child node
        for child in self.children:
            # Recursive call for the child node with the updated set of encountered nodes
            child_result = child.evaluate_constraints(encountered_nodes.copy(), result_feats)
        
            # Sum the values for each key in result_feats and child_result
            for key in result_feats:
                result_feats[key] += child_result.get(key, 0)

        return result_feats
    
    def merge_condition(self):
        merge_condition_results = {'merge_cond': 0}
        if len(self.children) > 1:
            if self.children[0].merge_feat and self.children[0].merge_feat != self.children[1].label:
                merge_condition_results['merge_cond'] += 1

            if self.children[1].merge_feat and self.children[1].merge_feat != self.children[0].label:
                merge_condition_results['merge_cond'] += 1 

        return merge_condition_results
    
# clone a node to avoid multiple assignment
def clone_tree(node):
    if node is None:
        return None
    
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

def traverse_and_clone(node, cloned_nodes, cloned_node_names=None):
    # Initialize cloned_node_names if not provided
    if cloned_node_names is None:
        cloned_node_names = set()

    # Base case: If the node is not None and has not already been cloned
    if node is not None and node.name not in cloned_node_names:
        # Clone the current node
        cloned_node = clone_tree(node)
        # Add the cloned node to the list of cloned nodes
        cloned_nodes.append(cloned_node)
        # Add the name of the cloned node to the set of cloned node names
        cloned_node_names.add(cloned_node.name)

    # Recursively traverse and clone the children of the current node
    for child in node.children or []:
        traverse_and_clone(child, cloned_nodes, cloned_node_names)

def parse_feats(feat_str):
    empty_dict = {} # for some f*cked up reason return {} throws an indentation error
    if feat_str is not None and feat_str.strip():
        return dict(pair.split(':') for pair in feat_str.split('-'))
    else: 
        return empty_dict

# read the csv file for nodes
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
    cloned_nodes = []

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
        new_1.merge_feat = {} # update this to check for merge_cond
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
    # Check if the node has sisters and no label
    if my_node.name or len(my_node.children) < 2:
        return []

    # Create a new node
    new_node = clone_tree(my_node)
    new_node.other_nodes = my_node.other_nodes

    # Agree left
    my_left_agr = my_node.children[0].agree_feats
    old_left_agr = my_node.children[0].agree_feats
    my_right_feats = my_node.children[1].neutral_feats
    for key, value in my_right_feats.items():
        if key + "_agr" in my_left_agr and value == my_left_agr[key + "_agr"]:
            my_left_agr[key + "_agr"] = 0

    new_node.children[0].agree_feats = my_left_agr

    # Agree right
    my_right_agr = my_node.children[1].agree_feats
    old_right_agr = my_node.children[1].agree_feats
    my_left_feats = my_node.children[0].neutral_feats
    for key, value in my_left_feats.items():
        if key + "_agr" in my_right_agr and value == my_right_agr[key + "_agr"]:
            my_right_agr[key + "_agr"] = 0

    # Only update the right child's agree_feats
    new_node.children[1].agree_feats = my_right_agr
    new_list = [new_node]
    return new_list

# function to form outputs from an input
def proceed_cycle(my_node):
    input_node = clone_tree(my_node)
    output_nodes = [].extend(Merge(input_node)).extend(Label(input_node)).extend(Agree(input_node))
    return output_nodes

# import numeration
my_nodes = read_nodes_csv("./unaccusative_numeration.csv")
my_result = Label(Merge(Label(Agree(Merge(Label(Merge(Label(Merge(Label(Merge(my_nodes[0])[0])[0])[0])[0])[0])[0])[4])[0])[1])[0])[0]

my_result.evaluate_constraints()

# Visualize the tree using ASCII art
for pre, _, node in RenderTree(my_result, style=AsciiStyle()):
    print(f"{pre}{node.name} - {node.agree_feats} - {node.neutral_feats} - {node.evaluate_constraints()} - {node.domination_count()}")

# Function to create labels with desired attributes
def node_label(node):
    return node.name

# Function to create unique names for DOT export
def unique_name(node):
    # Use a combination of node name and unique identifier
    return f'{node.name}_{id(node)}'

# export the dot_file
dot_filename = "tree.dot"
DotExporter(my_result, nodeattrfunc=lambda node: f'label="{node_label(node)}"', nodenamefunc=unique_name).to_dotfile(dot_filename)
# Convert DOT to SVG using Graphviz
svg_filename = "tree.svg"
os.system(f"dot -Tsvg {dot_filename} -o {svg_filename}")

# Define the PyQt5 application and main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the title and size of the main window
        self.setWindowTitle("Tree Visualization")
        self.setGeometry(100, 100, 400, 200)  # Adjusted main window size

        # Create a central widget and a layout for it
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a button to trigger tree display
        self.button = QPushButton("Click here to show the tree")
        self.button.clicked.connect(self.show_tree_popup)
        layout.addWidget(self.button)

    def show_tree_popup(self):
        # Create a QDialog for the tree popup
        dialog = QDialog(self)
        dialog.setWindowTitle("Tree Visualization")
        dialog.setGeometry(200, 200, 800, 600)  # Set the size and position of the dialog
        layout = QVBoxLayout(dialog)

        # Create a web view widget to display the graph
        graph_view = QWebEngineView(dialog)
        layout.addWidget(graph_view)

        # Load the SVG file and display it in the web view
        with open(svg_filename, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Display the SVG content in the web view
        graph_view.setHtml(svg_content)

        # Show the dialog
        dialog.exec_()

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()