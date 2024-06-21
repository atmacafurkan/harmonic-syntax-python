# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle
import copy
import csv
import pandas as pd
from typing import List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QMessageBox, QScrollArea, QSizePolicy, QApplication, QDialog, QTabWidget, QDoubleSpinBox
from PyQt5.QtSvg import QSvgWidget
from anytree.exporter import UniqueDotExporter
import sys
import os
from graphviz import Source
import tempfile
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
from pylatexenc.latexencode import unicode_to_latex
import re

agree_dict = {'case_agr': 0, 'wh_agr': 0, 'foc_agr': 0, 'cl_agr':0}
neutral_dict = {'case': 0, 'wh': 0, 'foc' : 0,'cl':0}
empty_dict = {'case_mt': 0, 'wh_mt': 0, 'foc_mt': 0, 'cl_mt':0}
used_feats_dict = neutral_dict
constraints_dict = {'merge_cond' : 0, 'exhaust_ws': 0, 'label_cons': 0, **agree_dict, **neutral_dict, **empty_dict}
explanations_dict = {'input':'The input for the derivation cycle',
                     'winner':'The optimal output of the derivation cycle',
                     'operation': 'The operation name given for easy intrepretation. xMerge, iMerge, and rMerge are all one operation Merge.',
                     'output': 'A linear representation of the output. You can click on it to view the visual representation.',
                     'merge_cond': 'Merge condition constraint, a constraint tied to the operation Merge. It is violated when the merge feature of one item does not match the label of the other.',
                     'label_cons': 'Labelling constraint, a constraint tied to the operation Merge. It is violated when the result of the merge does not have a label.',
                     'case_agr': 'Markedness constraint for case agreement feature.',
                     'wh_agr': 'Markedness constraint for wh agreement feature.',
                     'foc_agr': 'Markedness constraint for focus agreement feature.',
                     'cl_mt': 'Markedness constraint for classifier agreement feature.',
                     'case': 'Markedness constraint for case feature.',
                     'wh': 'Markedness constraint for wh feature.',
                     'foc': 'Markedness constraint for focus feature.',
                     'cl': 'Markedness constraint for classifier feature',
                     'case_mt': 'Empty agreement constraint for case, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.',
                     'foc_mt': 'Empty agreement constraint for focus, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.',
                     'wh_mt': 'Empty agreement constraint for wh, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.',
                     'cl_mt': 'Empty agreement constraint for classifier, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.'
                     }

# %%
# Section: Functions for weight optimization
# Kullback-Leibler divergence 
def KL(p, q): 
    mask = p != 0  # Create a mask to avoid log(0)
    kl_divergence = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return kl_divergence

def KL_divergence(p, q):
    """
    Compute Kullback-Leibler (KL) divergence between two distributions p and q.

    Parameters:
    p, q : array-like
        Arrays representing the distributions. They must have the same shape.

    Returns:
    float
        KL divergence value.
    """
    # Convert inputs to numpy arrays to ensure compatibility
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ensure distributions sum to 1 (normalization)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Mask to handle cases where p[i] == 0
    mask = p != 0

    # Compute KL divergence
    kl_divergence = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return kl_divergence

def frequency_and_probabilities(x, df, update_table = False):
    # get weights
    constraint_weights = x[np.newaxis,:]

    # new df
    new_df = df.copy()

    divergences = []
    for name, group in df.groupby('input'):
        # Convert the group to a matrix (2D numpy array)
        matrix = group.drop(columns = ['input']).to_numpy(dtype=np.float64)

        # Slice the matrix to exclude the first column
        matrix_to_multiply = matrix[:, 1:]

        # Multiply each row of the sliced matrix by corresponding elements of x
        multiplied = matrix_to_multiply * (-constraint_weights)

        # Sum all values in each row
        harmony_values = np.sum(multiplied, axis=1, keepdims=True)

        # Compute softmax of row_sums
        probabilities = np.exp(harmony_values)/np.sum(np.exp(harmony_values))

        # get the frequencies from the winner
        frequencies = matrix[:,0].astype(np.float64)

        if update_table: # only do this when updating the table, not when calculating optimization.
            # Add probabilities and harmony values to the group
            group['probabilities'] = probabilities.flatten()
            group['harmony_score'] = harmony_values.flatten()

            # Append the modified group back to the new DataFrame
            new_df.loc[group.index, 'probabilities'] = group['probabilities']
            new_df.loc[group.index, 'harmony_score'] = group['harmony_score']

        # calculate divergence
        divergences.append((frequencies,probabilities))

    return divergences, new_df

# optimizing function
def weight_optimize(my_tableaux):
    # Reorder the DataFrame
    df_reordered = reorder_table(my_tableaux)

    # group the cumulative eval in a data frame
    df = df_reordered.drop(columns = ['output', 'operation'])
    
    # number of constraints
    n_constraint = len(df.columns) - 2

    # Initial guess for the weights
    initial_weights = np.zeros(n_constraint)
    
    # Define the bounds for the weights (0 to 200)
    bounds = [(0, 100) for _ in range(n_constraint)]

    # Define the objective function that takes only the weights as input
    def objective(weights):
        my_divergences, _ = frequency_and_probabilities(weights, df)
        total_KL_divergence = sum(KL_divergence(freq, prob) for freq, prob in my_divergences)
        return total_KL_divergence ** 2
    
    # Perform the optimization using L-BFGS-B method
    result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=bounds)
    
    # Return the resulting optimization
    return result

# Custom node class with a named list field
class SyntaxNode(Node):
    def __init__(self, name, label = None, merge_feat = None, neutral_feats = None, empty_agr = None, result_feats = None, agree_feats = None, used_feats = None, other_nodes = None, operation = None, exhaust_ws = None, parent = None, children = None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else None
        self.merge_feat = merge_feat if merge_feat is not None and merge_feat != '' else None
        self.agree_feats = agree_feats if agree_feats is not None else agree_dict
        self.neutral_feats = neutral_feats if neutral_feats is not None else neutral_dict
        self.used_feats = used_feats if used_feats is not None else used_feats_dict
        self.empty_agr = empty_agr if empty_agr is not None else empty_dict
        self.result_feats = result_feats if result_feats is not None else constraints_dict
        self.operation = operation if operation is not None else None
        self.exhaust_ws = exhaust_ws if exhaust_ws is not None else None
        self.other_nodes = other_nodes if other_nodes is not None else []

    def add_other_node(self, other_node):
        self.other_nodes.append(other_node)
            
    def domination_count(self):
        # Get all distinct parent labels recursively
        parent_labels = set()
        current_parent = self.parent

        while current_parent:
            if current_parent.name and current_parent.name != self.name:
                parent_labels.add(current_parent.name)
            elif current_parent.name == self.name: # if the parent label and the current label are the same
                return 0
            current_parent = current_parent.parent
        # Return the count of distinct parent labels
        return len(parent_labels)

    def evaluate_constraints(self, encountered_nodes=None, default_state = None):
        # Initialize a result dictionary for both agree_feats and neutral_feats
        result_feats = constraints_dict.copy() if default_state is None else default_state

        # Initialize encountered_nodes if not provided
        encountered_nodes = set() if encountered_nodes is None else encountered_nodes

        # If encountered before, set domination count to 0
        if self.name in encountered_nodes:
            domination_count = 0
        else:
            domination_count = self.domination_count()
            encountered_nodes.add(self.name)

        # Update feature violations
        for key, value in {**self.agree_feats, **self.neutral_feats}.items():
            result_feats[key] += int(value) * domination_count     

        # Recursive call for each child node
        if len(self.children):
            # go left
            self.children[0].evaluate_constraints(encountered_nodes, result_feats)

            # go right
            self.children[1].evaluate_constraints(encountered_nodes, result_feats)

        return result_feats
  
# function to draw linear representation of the tree
    def to_linear_ex(self):
        if not self:
            return ""
        
        if self.name:
            my_name = str(self.name)
        else:
            my_name = ".."

        # add agree features
        agree_feats = [key for key, value in self.agree_feats.items() if value == "1"]
        if agree_feats:
            my_name += " " + ",".join(agree_feats)

        # add neutral features
        neutral_feats = [key for key, value in self.neutral_feats.items() if value == "1"]
        if neutral_feats:
            my_name += " " + ",".join(neutral_feats)
                
        result = "[" + my_name
        if self.children:
            result += " " + " ".join(child.to_linear() for child in self.children)
        result += "]"
    
        return result

    # function to draw linear representation of the tree
    def to_linear(self):
        if not self:
            return ""
    
        if self.name:
            my_name = str(self.name)
        else:
            my_name = ".."

        # add agree features
        agree_feats = [key for key, value in self.agree_feats.items() if value == "1"]
        # add neutral features
        neutral_feats = [key for key, value in self.neutral_feats.items() if value == "1"]

        # combine agree and neutral features and join them with commas
        combined_feats = agree_feats + neutral_feats
        if combined_feats:
            my_name += " " + ",".join(combined_feats)
            
        result = "[" + my_name
        if self.children:
            result += " " + " ".join(child.to_linear() for child in self.children)
        result += "]"

        return result        

# merge condition, only checked at the root node (latest operation)
def merge_condition(node):
    violation = 0
    if len(node.children) > 1:
        if (node.children[0].merge_feat and node.children[0].merge_feat != node.children[1].name) or not node.children[0].name:
            violation += 1

        if (node.children[1].merge_feat and node.children[1].merge_feat != node.children[0].name) or not node.children[1].name:
            violation += 1

    return violation

# for label constraint, only checked at the root node (latest operation)
def label_constraint(node):
    # if the node does not have a name and the operation is Merge, return 1
    if not node.name and node.operation in ["xMerge","iMerge","rMerge"]:
        return 1
    else:
        return 0

# for empty agreement, only checked at the root node (since Agree would be the latest operation)
def empty_agreement(node, result_dict):
    empty_dict = node.empty_agr
    for key, value in empty_dict.items():
        result_dict[key] = int(value)
    return result_dict
    
# clone a node to avoid multiple assignment
def clone_tree(node):
    if node is None:
        return None
    else:
        cloned_node = copy.deepcopy(node)
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
                merge_feat= str(row['mc']),
                agree_feats=parse_feats(row['ac']),
                neutral_feats=parse_feats(row['ft']),
                empty_agr = None
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
        if original_node.name == new_right.name:
            new_node.name = original_node.name

        new_node.operation = "xMerge"
        new_node.exhaust_ws = 0
        output_nodes.append(new_node)

    # Separate loop over cloned nodes
    # internal merge
    for cloned_node in cloned_nodes[1:]:
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
        if new_left.name == new_right.name:
            new_node.name = new_left.name
        
        new_node.operation = "iMerge"
        new_node.exhaust_ws = 1
        output_nodes.append(new_node)

    # add the reflexive merge as the final candidate
    reflexive_merge = clone_tree(my_arg)
    reflexive_merge.operation = "rMerge"
    reflexive_merge.exhaust_ws = 1
    output_nodes.append(reflexive_merge)

    return output_nodes

# Label function
def Label(my_node):
    my_nodes = []
    if not my_node.name and len(my_node.children) == 2:
        # take from left
        new_1 = clone_tree(my_node)
        new_1.merge_feat = {} # update this to check for merge_cond
        new_1.name = new_1.children[0].name
        new_1.agree_feats = new_1.children[0].agree_feats

        # check agreement
        new_1_neutral = new_1.children[0].neutral_feats.copy()  # Make a copy to modify
        for key, value in new_1_neutral.items():
            if value == "1" and new_1.children[0].used_feats[key] == "1":
                new_1_neutral[key] = "0"  # Reset to 0
        new_1.neutral_feats = new_1_neutral

        new_1.operation = "Label"
        new_1.exhaust_ws = 0
        my_nodes.append(new_1)

        # take from right
        new_2 = clone_tree(my_node)
        new_2.merge_feat = {}
        new_2.name = new_2.children[1].name
        new_2.agree_feats = new_2.children[1].agree_feats

        new_2_neutral = new_2.children[1].neutral_feats.copy()  # Make a copy to modify
        for key, value in new_2_neutral.items():
            if value == "1" and new_2.children[1].used_feats[key] == "1":
                new_2_neutral[key] = "0"  # Reset to 0
        new_2.neutral_feats = new_2_neutral

        new_2.neutral_feats = new_2_neutral
        new_2.operation = "Label"
        new_2.exhaust_ws = 0
        my_nodes.append(new_2)
    return(my_nodes)

# Agree function, only under sisterhood
def Agree(my_node):
    new_list = []    
    # empty agreement if it is a labelled node and has agreement features
    if len(my_node.name) > 0 and "1" in my_node.agree_feats.values():
        new_node = clone_tree(my_node)

        my_agr = my_node.agree_feats.copy()  # Create a copy to avoid modifying the original node's attributes
        my_empty = my_node.empty_agr.copy()
        for key, value in my_agr.items():
            if my_agr[key] == '1':
                my_agr[key] = 0
                my_empty[key.replace('_agr', '') + '_mt'] = 1
        
        new_node.agree_feats = my_agr
        new_node.empty_agr = my_empty
        new_node.operation = "mtAgree"
        new_node.exhaust_ws = 0
        if my_node.agree_feats != new_node.agree_feats:
            new_list.append(new_node)
                  
    if len(my_node.children) > 0 and len(my_node.name) == 0:
        # Create a new node
        newer_node = clone_tree(my_node)
        newer_node.other_nodes = my_node.other_nodes

        # Preserve original agree_feats for comparison
        original_left_agr = copy.deepcopy(my_node.children[0].agree_feats)
        original_right_agr = copy.deepcopy(my_node.children[1].agree_feats)

        # Agree left
        my_left_agr = my_node.children[0].agree_feats
        my_right_feats = my_node.children[1].neutral_feats
        my_used_right = my_node.children[1].used_feats
        for key, value in my_right_feats.items():
            if key + "_agr" in my_left_agr and value == my_left_agr[key + "_agr"]:
                my_left_agr[key + "_agr"] = "0"
                if value == "1":
                    my_used_right[key] = "1"
                
        newer_node.children[0].agree_feats = my_left_agr
        newer_node.children[1].used_feats = my_used_right

        # Agree right
        my_right_agr = my_node.children[1].agree_feats
        my_left_feats = my_node.children[0].neutral_feats
        my_used_left = my_node.children[0].used_feats

        for key, value in my_left_feats.items():
            if key + "_agr" in my_right_agr and value == my_right_agr[key + "_agr"]:
                my_right_agr[key + "_agr"] = "0"
                if value == "1":
                    my_used_left[key] = "1"

        # Only update the right child's agree_feats
        newer_node.children[1].agree_feats = my_right_agr
        newer_node.children[0].used_feats = my_used_left

        newer_node.operation = "Agree"
        newer_node.exhaust_ws = 0

         # Check if there were changes in agree_feats
        if original_left_agr != my_left_agr or original_right_agr != my_right_agr:
            new_list.append(newer_node)
    return new_list

# function to form outputs from an input
def proceed_cycle(input_node):
    output_nodes = []

    for_agree = copy.deepcopy(input_node)
    output_nodes.extend(Agree(for_agree)) # carry out agree

    for_label = copy.deepcopy(input_node)
    output_nodes.extend(Label(for_label)) # carry out label

    for_merge = copy.deepcopy(input_node)
    output_nodes.extend(Merge(for_merge)) # carry out merge

    return output_nodes

# Function to generate SVG content in memory
def generate_svg_content(root_node):
    # Create a UniqueDotExporter
    with tempfile.NamedTemporaryFile(delete=False) as tmp_dot_file:
        # Export DOT data to the temporary file
        UniqueDotExporter(root_node).to_dotfile(tmp_dot_file.name)

        # Read DOT data from the temporary file
        with open(tmp_dot_file.name, 'r') as f:
            dot_data = f.read()

    # Convert DOT to SVG in memory using Graphviz
    src = Source(dot_data)
    svg_bytes = src.pipe(format='svg')
    svg_content = svg_bytes.decode('utf-8')
    
    return svg_content

# function to generate data frame fro widget
def table_to_dataframe(table_widget):
    # Initialize a list to store the data
    data = []

    # Loop over all rows in the table
    for row in range(table_widget.rowCount()):
        # Initialize a list to store the row data
        row_data = []
        # Loop over all columns in the table
        for column in range(table_widget.columnCount()):
            # Get the QTableWidgetItem for the cell
            item = table_widget.item(row, column)
            # Check if the item is not None and append its text, otherwise append an empty string
            row_data.append(item.text() if item is not None else '')
        # Append the row data to the main data list
        data.append(row_data)

    # Get the column headers
    headers = []
    for column in range(table_widget.columnCount()):
        header_item = table_widget.horizontalHeaderItem(column)
        headers.append(header_item.text() if header_item is not None else '')

    # Create a pandas DataFrame from the data and set the column headers
    df = pd.DataFrame(data, columns=headers)
    df.reset_index(drop=True, inplace = True)
    return df

def reorder_table(df, columns_to_move = ['input','winner','operation','output']):
    # order the data frame
    remaining_columns = [col for col in df.columns if col not in columns_to_move]

    # Create the new column order
    new_order = columns_to_move + remaining_columns
    
    # Reorder the DataFrame
    return df[new_order]

# Define the PyQt5 application and main window
# %%
# Section: GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Harmonic Syntax Tabulator')
        self.initUI()
        
    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)
        self.createTabs()   

    def createTabs(self):
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        self.tabs.addTab(self.tab1, "Make Derivations")
        self.tabs.addTab(self.tab2, "Find weights for multiple evaluations")
        self.tabs.addTab(self.tab3, "Export derivations and cycles as latex tables")
        self.tabs.addTab(self.tab4, "Test markedness bias")

        self.addContentTab1()
        self.addContentTab2()
        self.addContentTab3()
        self.addContentTab4()

# %%
# Section: The first tab
    def addContentTab1(self):
        # main layout
        mainLayout = QHBoxLayout()

        left_side = QVBoxLayout()
        right_side = QVBoxLayout()

        # selecting the numeration
        self.label_optimal = QLabel("Double Click on the row name to select it as the optimal output.")
        self.select_numeration = QPushButton('Select Numeration')
        self.select_numeration.clicked.connect(self.import_numeration)

        # export the cumulative eval
        self.eval_export = QPushButton('Export the cumulative eval')
        self.eval_export.setEnabled(False)
        self.eval_export.clicked.connect(self.export_eval)

        # export the derivation with weights
        self.der_export = QPushButton('Export the cumulative eval with weights')
        self.der_export.setEnabled(False)
        self.der_export.clicked.connect(lambda: self.export_derivation(self.table_eval))

        # run the weight optimizer
        self.find_weights = QPushButton('Run the weight optimizer')
        self.find_weights.setEnabled(False)
        self.find_weights.clicked.connect(lambda: self.run_minimazing_KL(self.cumulative_eval, self.table_eval))
        
        # displaying the input tree
        self.input_tree = QSvgWidget(self)
        self.input_tree.setFixedSize(500, 700)  # Set fixed size
        self.input_tree.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        left_side.addWidget(self.input_tree)
        left_side.addWidget(self.select_numeration)
        left_side.addWidget(self.label_optimal)
        left_side.addWidget(self.eval_export)
        left_side.addWidget(self.find_weights)
        left_side.addWidget(self.der_export)

        # displaying eval
        # Create a QTableWidget
        self.table_eval = QTableWidget(self)
        self.table_eval.setColumnCount(len(constraints_dict) + 1)
        self.table_eval.cellClicked.connect(self.on_cell_clicked) # when the output is clicked, port to tree visualisation
        self.table_eval.horizontalHeader().sectionClicked.connect(self.on_header_clicked) # when the column names are clicked, connect to explanations
        self.table_eval.verticalHeader().sectionDoubleClicked.connect(self.next_cycle)# when the rows are clicked, connect to proceed cycle

        right_side.addWidget(self.table_eval)

        mainLayout.addLayout(left_side)
        mainLayout.addLayout(right_side)

        self.tab1.setLayout(mainLayout)

    def export_derivation(self, the_table):
        # Open a file dialog to ask for file name and location to save
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save the derivation with weights", "", "CSV Files (*.csv)", options=options)
        if file_name:
            # Ensure the file has a .csv extension
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            self.save_table_as_csv(file_name, the_table)

    def save_table_as_csv(self, file_name, the_table):
        # Save the table data as a CSV file
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the headers
            headers = [the_table.horizontalHeaderItem(col).text() for col in range(the_table.columnCount())]
            writer.writerow(headers)
            # Write the data
            for row in range(the_table.rowCount()):
                row_data = [the_table.item(row, col).text() if the_table.item(row, col) else '' for col in range(the_table.columnCount())]
                writer.writerow(row_data)

    def export_eval(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save the derivation with weights", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.cumulative_eval = reorder_table(self.cumulative_eval)
            # Ensure the file has a .csv extension
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            self.cumulative_eval.to_csv(file_name, index=False)
        
        self.label_optimal.setText('Eval exported!')
        self.label_optimal.setStyleSheet('color : green')

    def import_numeration(self):
        # Clear the table
        self.table_eval.clear()
        self.table_eval.setRowCount(0)
        self.table_eval.setColumnCount(0)

        # Clear my eval
        self.my_eval = None

        # create empty eval table
        my_columns = list(constraints_dict.keys()) + ['input', 'winner']

        self.cumulative_eval = pd.DataFrame(columns = my_columns)

        # enable header and output selection
        self.cycle_enabled = True

        # disable weight optimizer
        self.find_weights.setEnabled(False)

        # disable derivation export
        self.der_export.setEnabled(False)

        self.numeration_path, _ = QFileDialog.getOpenFileName(self, 'Select Numeration', '.', 'Csv Files (*.csv)')
        self.numeration = read_nodes_csv(self.numeration_path)

        # update headers once for labelling constraints
        self.available_names = None
        self.available_names = [node.name for node in self.numeration]
        
        self.headers = list(constraints_dict.keys()) # get the keys from constraints dict and available names
        self.headers += ['LB_' + s for s in self.available_names] 

        if self.numeration_path:
            # update the input tree
            my_input = generate_svg_content(self.numeration[0])

            # update the input
            self.the_input = copy.deepcopy(self.numeration[0])

            # Load the SVG into QSvgWidget
            self.input_tree.load(my_input.encode('utf-8'))

            # run the first cycle for the input
            self.outputs = proceed_cycle(self.numeration[0])

            # Update the evaluation
            self.update_eval()

    def update_eval(self):
        # Clear the table
        self.clear_table_widget(self.table_eval)

        # Set the number of rows in the table
        self.table_eval.setRowCount(len(self.outputs))

        # Set the headers for the table
        headers = ['operation'] + ['output'] + self.headers

        # Set the number of columns in the table
        self.table_eval.setColumnCount(len(headers))
        
        # set new headers
        self.table_eval.setHorizontalHeaderLabels(headers)

        # Ensure all columns are visible
        for col in range(self.table_eval.columnCount()):
            self.table_eval.setColumnHidden(col, False)
    
        # Populate the table
        for row, node in enumerate(self.outputs):
            # Populate the first column with the node name
            name_operation = QTableWidgetItem(node.operation)
            name_operation.setFlags(name_operation.flags() & ~Qt.ItemIsEditable)  # Make the item non-editable

            self.table_eval.setItem(row, 0, name_operation)

            name_item = QTableWidgetItem(node.to_linear())  # Assuming node.name is the attribute for the node's name
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table_eval.setItem(row, 1, name_item)
            
            new_node = clone_tree(node)
            data_dict = new_node.evaluate_constraints()
            data_dict['merge_cond'] = merge_condition(new_node) # check for merge condition
            data_dict['label_cons'] = label_constraint(new_node) # check for label constraint
            data_dict['exhaust_ws'] = new_node.exhaust_ws #exhaust workspace constraint
            data_dict = empty_agreement(new_node, data_dict) # check for empty agreement

            if new_node.operation == "Label": # labelling constraint
                data_dict['LB_' + new_node.name] = 1
            
            for col, key in enumerate(self.headers, start=2):  # Start from column 3 to skip the first two columns
                value = data_dict.get(key, 0)
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table_eval.setItem(row, col, item)

        # copy table_eval for export purposes (since unused constraints are later omitted, it might cause problems)
        self.my_eval = table_to_dataframe(self.table_eval)

        # Check for columns to hide
        columns_to_hide = []
        for col in range(2, self.table_eval.columnCount()):  # Start from the 3rd column
            column_has_1 = any(self.table_eval.item(row, col).text() == "1" for row in range(self.table_eval.rowCount()))
            if not column_has_1:
                columns_to_hide.append(col)
        
        # Hide columns that should be hidden
        for col in columns_to_hide:
            self.table_eval.setColumnHidden(col, True)

        # Resize columns to content
        self.table_eval.resizeColumnsToContents()

    def next_cycle(self, logicalIndex):
        if self.cycle_enabled == False:
            return None
        
        # save the eval table to a data frame
        eval_df = self.my_eval
        eval_df['input'] = self.the_input.to_linear()
        eval_df['winner'] = 0
        eval_df.loc[logicalIndex, 'winner'] = 1

        # enable eval export
        self.eval_export.setEnabled(True)

        # update cumulative eval
        self.cumulative_eval = pd.concat([self.cumulative_eval, eval_df], axis=0, ignore_index=True)

        # get the selected output
        selected_output = self.outputs[logicalIndex]

        # check derivation convergence
        if selected_output.operation == "rMerge":
            self.display_derivation()
        else: 
            # update the input
            self.the_input = copy.deepcopy(selected_output)
        
            # produce the new outputs
            self.outputs = proceed_cycle(selected_output)

            # update the input tree
            my_input = generate_svg_content(selected_output)

            # Load the SVG into QSvgWidget
            self.input_tree.load(my_input.encode('utf-8'))

            # update the evaluation table
            self.update_eval()

    def display_derivation(self):
        # Clear the table
        self.clear_table_widget(self.table_eval)

        # get the cumulative eval
        my_tableaux = self.cumulative_eval

        # order the data frame
        columns_to_move = ['input','winner','operation','output']
        remaining_columns = [col for col in my_tableaux.columns if col not in columns_to_move]

        # Create the new column order
        new_order = columns_to_move + remaining_columns
    
        # Reorder the DataFrame
        df = my_tableaux[new_order]

        # Set column count
        self.table_eval.setColumnCount(len(df.columns))

        # Set row count
        self.table_eval.setRowCount(len(df.index))
        
        # Set the table headers
        self.table_eval.setHorizontalHeaderLabels(df.columns)
        
        # Populate the table with data
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iat[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make cell not editable
                self.table_eval.setItem(i, j, item)

        # Resize columns to content
        self.table_eval.resizeColumnsToContents()

        # disable header or output selection
        self.cycle_enabled = False

        # enable weight optimizer
        self.find_weights.setEnabled(True)

    def on_cell_clicked(self, row, column):
        # Custom function to be executed when a cell is clicked
        if column == 1 and self.cycle_enabled == True:
            svg_content = generate_svg_content(self.outputs[row])

            # Create a clickable popup dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Tree Visualization")

            # Create a QSvgWidget to display the graph
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(svg_content.encode('utf-8'))

            # Optionally add scroll area for large SVGs
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(svg_widget)

            # Create a QTableWidget to display the node attributes
            table_widget = QTableWidget()
            table_widget.setColumnCount(7)
            table_widget.setHorizontalHeaderLabels(['node', 'name', 'agr_feats', 'neut_feats', 'dominators', 'used_feats', 'operation'])

            # Fetch tree structure and node attributes
            root_node = self.outputs[row]
            for pre, _, node in RenderTree(root_node, style=AsciiStyle()):
                row_position = table_widget.rowCount()
                table_widget.insertRow(row_position)
                table_widget.setItem(row_position, 0, QTableWidgetItem(pre + node.name))
                table_widget.setItem(row_position, 1, QTableWidgetItem(node.name))
                table_widget.setItem(row_position, 2, QTableWidgetItem(str(node.agree_feats)))
                table_widget.setItem(row_position, 3, QTableWidgetItem(str(node.neutral_feats)))
                table_widget.setItem(row_position, 4, QTableWidgetItem(str(node.domination_count())))
                table_widget.setItem(row_position, 5, QTableWidgetItem(str(node.used_feats)))
                table_widget.setItem(row_position, 6, QTableWidgetItem(str(node.operation)))

            # Resize columns to content
            table_widget.resizeColumnsToContents()
            # Set size policy and layout
            svg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout = QVBoxLayout(dialog)
            layout.addWidget(scroll_area)
            layout.addWidget(table_widget)

            # Resize the dialog based on SVG dimensions
            dialog.resize(800,800)
            dialog.setLayout(layout)

            # Show the dialog
            dialog.show()

    def on_header_clicked(self, logicalIndex):
        if self.cycle_enabled == True:
            header_label = self.table_eval.horizontalHeaderItem(logicalIndex).text()
            message = f"{explanations_dict[header_label]}"
            self.show_message(message)

    def show_message(self, message):
        # Show a message box with the given message
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(message)
        msgBox.setWindowTitle("Explanation")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()

# %%
# Section: The second tab      
    def addContentTab2(self):
        layout = QVBoxLayout()
        self.tab2.setLayout(layout)

        # Concatenate button
        self.concatButton = QPushButton('Select and Concatenate Evaluations')
        self.concatButton.clicked.connect(lambda: self.concatenate_csv(self.table_combined_eval))
        layout.addWidget(self.concatButton)

        # Concatenate button
        self.find_weights2 = QPushButton('Find constraint weights')
        self.find_weights2.setEnabled(False)
        self.find_weights2.clicked.connect(lambda: self.run_minimazing_KL(self.combined_cumulative_eval,self.table_combined_eval))
        layout.addWidget(self.find_weights2)

        # export the derivation with weights
        self.der_export2 = QPushButton('Export the cumulative eval with weights')
        self.der_export2.setEnabled(False)
        self.der_export2.clicked.connect(lambda: self.export_derivation(self.table_combined_eval))
        layout.addWidget(self.der_export2)

        # Table to display concatenated data
        self.table_combined_eval = QTableWidget()
        layout.addWidget(self.table_combined_eval)

    def concatenate_csv(self, the_table):
        # Open file dialog to select multiple CSV files
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)", options=options)
        if files:
            self.import_evals = files

            try:
                # List to hold the DataFrames
                dfs = []

                # Read each CSV file into a DataFrame
                for csv_file in self.import_evals:
                    df = pd.read_csv(csv_file)

                    file_name = os.path.basename(csv_file)
                    prefix = file_name.split('_')[0]

                    # Append the prefix to the 'input' column
                    if 'input' in df.columns:
                        df['input'] = prefix + "_" + df['input'].astype(str)

                    dfs.append(df)

                # Concatenate all DataFrames in the list
                concatenated_df = pd.concat(dfs, ignore_index=True)

                # Fill NaN values with 0
                concatenated_df = concatenated_df.fillna(0)

                self.combined_cumulative_eval = concatenated_df

                # Display concatenated data in the table
                self.display_combined_eval(concatenated_df, the_table)

            except Exception as e:
                QMessageBox.critical(self, 'Error in combining csv files', str(e))

    def display_combined_eval(self, df, the_table):
        # Clear existing table
        the_table.clear()

        # Set number of rows and columns
        the_table.setRowCount(df.shape[0])
        the_table.setColumnCount(df.shape[1])

        # Set headers
        the_table.setHorizontalHeaderLabels(df.columns.astype(str))

        # Populate table with data
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                the_table.setItem(i, j, item)

        # Resize columns to content
        the_table.resizeColumnsToContents()

        self.find_weights2.setEnabled(True)

    def clear_table_widget(self, table_widget):
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0) 

    def run_minimazing_KL(self, the_eval, the_table):
        self.optimization = weight_optimize(the_eval)
        headers = [the_table.horizontalHeaderItem(i).text() for i in range(the_table.columnCount())]
        # Update headers starting from the specified column index
        for i, value in enumerate(self.optimization.x, start=4):
            if i < len(headers):
                headers[i] += f"_({value:.2f})"

        # Set updated headers back to the table
        the_table.setHorizontalHeaderLabels(headers)

        # update the table with the new probabilities
        my_solution = table_to_dataframe(the_table) 

        # add the probabilities
        _, new_solution = frequency_and_probabilities(self.optimization.x, reorder_table(my_solution).drop(columns = ['output', 'operation']), update_table= True)

        probabilities = new_solution['probabilities']
        harmonies = new_solution['harmony_score']
        
        the_table.insertColumn(2)
        the_table.setHorizontalHeaderItem(2, QTableWidgetItem('probability'))

        the_table.insertColumn(3)
        the_table.setHorizontalHeaderItem(3, QTableWidgetItem('harmony'))

        for row in range(len(probabilities)):
            my_probability = QTableWidgetItem(str(probabilities[row]))
            my_probability.setFlags(my_probability.flags() & ~Qt.ItemIsEditable)
            the_table.setItem(row, 2, my_probability)

            my_harmony = QTableWidgetItem(str(harmonies[row]))
            my_harmony.setFlags(my_harmony.flags() & ~Qt.ItemIsEditable)
            the_table.setItem(row, 3, my_harmony)

        # Resize columns to content
        the_table.resizeColumnsToContents()

        # enable derivation export
        self.der_export.setEnabled(True)
        self.der_export2.setEnabled(True)

    def addContentTab3(self):
        layout = QVBoxLayout()
        self.tab3.setLayout(layout)

        # Concatenate button
        self.read_der = QPushButton('Select the derivation')
        self.read_der.clicked.connect(lambda: self.read_derivation_csv(self.table_derivation))
        layout.addWidget(self.read_der)

        # export latex tables 
        self.export_latex = QPushButton('Export latex tables')
        self.export_latex.clicked.connect(self.write_latex_tables)
        layout.addWidget(self.export_latex)

        self.label_latex = QLabel("")
        layout.addWidget(self.label_latex)

        # Table to display the imported derivation
        self.table_derivation = QTableWidget()
        layout.addWidget(self.table_derivation)

    def read_derivation_csv(self, the_table):
        # Open file dialog to select a single CSV file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if file:
            df = pd.read_csv(file)

            # Fill NaN values with 0
            df = df.fillna(0)

            self.combined_cumulative_eval = df

            # Display concatenated data in the table
            self.display_combined_eval(df, the_table)

            self.label_latex.setText(f'Type of data for probability: {df['probability'].dtype}')
            self.label_latex.setStyleSheet('color : blue')

    # Function to convert a DataFrame to a LaTeX table
    def df_to_latex(self, df, input_value):
        table_header = f'Input={unicode_to_latex(input_value)}'

        # drop unused columns and only keep violated constraints
        dx = df.drop(columns=['input','harmony','derivation'])
        dx = dx.loc[:, (dx != 0).any(axis=0)]

        # rename columns for reducing size
        latex_names = {'merge_cond': 'mc', 'exhaust_ws': 'xws', 'label_cons': 'lab', 'operation': 'opr.', 'winner': 'W', 'probability': 'prb.', 'LB' : 'lb'}

        # Iterate over columns and rename based on the dictionary
        for col in dx.columns:
            for pattern, replacement in latex_names.items():
                if re.search(pattern, col):
                    new_col = re.sub(pattern, replacement, col)
                    dx.rename(columns={col: new_col}, inplace=True)
                    break

        # format the output
        # Generate LaTeX-formatted table
        table = tabulate(dx, headers='keys', tablefmt='latex', showindex=False, floatfmt=".2f")
        # Replace "{rrll" with "{rrlX" so that output can only extend the table upto the linewidth limit
        table = re.sub(r'\{rrll', '{\\\\linewidth}{rrlX', table)
        # replace weights with superscripts
        table = re.sub(r"\\_\((\d+\.\d+)\)", r"$^{\1}$", table)
        # replace substring names in constraints
        table = re.sub(r"\\_([a-zA-Z0-9]+)\$\^\{([0-9\.]+)\}\$", r"$_{\1}^{\2}$", table)
        # replace input, output substrings
        table = re.sub(r"\\_([a-zA-Z0-9]+)", r"_{\1}", table)
        table = re.sub(r'(\[\w+)\s([^\[\]\s]+)([\s\]])', r"\1$_{\2}$\3", table)

        table_header = re.sub(r"\\_([a-zA-Z0-9]+)", r"_{\1}", table_header)
        table_header = re.sub(r'(\[\w+)\s([^\[\]\s]+)([\s\]])', r"\1$_{\2}$\3", table_header)
        
        # remove .00 
        table = table.replace(".00", "")
        # replace 0. with .
        table = table.replace("0.", ".")
        # replace tabular environment
        table = table.replace("tabular", "tabularx")

        return "\\begingroup\\scriptsize " + table_header + "\\\\*\n" + table + "\\endgroup\\\\" + '\n'
    
    def write_latex_tables(self):
        df = self.combined_cumulative_eval
        df[['derivation', 'input']] = df['input'].str.split('_', n = 1, expand=True)

        # Get unique values of derivation and input in the order they appear
        derivation_order = df['derivation'].unique()
        
        # Concatenate all tables into a single string
        all_tables = ''
        for my_der in derivation_order:
            group1 = df[df['derivation'] == my_der]
            input_order = group1['input'].unique()
            all_tables += f'\\subsection{{The derivation={my_der}}}\n'
            for my_input in input_order:
                group2 = group1[group1['input'] == my_input]
                if not group2.empty:
                    all_tables += self.df_to_latex(group2, my_input)
            
        # Use QFileDialog to get the file path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save LaTeX File As",
            "",
            "LaTeX files (*.tex)",
            options=options
        )

        if file_path:
            # Write the concatenated LaTeX tables to the chosen .tex file
            with open(file_path, 'w') as f:
                f.write(all_tables)
            self.set_text_color(self.label_latex, "Derivation tables are exported!", "green")
        else:
            self.set_text_color(self.label_latex, "Export cancelled or no file selected.", "red")

    def set_text_color(self, the_label, the_text, the_color):
        the_label.setText(the_text)
        the_label.setStyleSheet(f'color : {the_color}')

    def addContentTab4(self):
        layout = QVBoxLayout()
        self.tab4.setLayout(layout)

        # Concatenate button
        self.concatButton = QPushButton('Select and Concatenate Evaluations')
        self.concatButton.clicked.connect(lambda: self.concatenate_csv(self.table_combined_eval))
        layout.addWidget(self.concatButton)

        # select the amount of markedness bias
        self.label = QLabel("Enter an amount for markedness bias:")
        layout.addWidget(self.label)

        # select the amount of markedness bias
        self.markedness_bias = QDoubleSpinBox()
        self.markedness_bias.setDecimals(2)    # Set number of decimal places
        self.markedness_bias.setRange(0.0, 100.0) # Set range
        self.markedness_bias.setSingleStep(0.1) # Set step size
        self.markedness_bias.setValue(0.0) 
        layout.addWidget(self.markedness_bias)

        # Concatenate button
        self.apply_markedness_bias = QPushButton('Find weights and apply the markedness bias')
        self.apply_markedness_bias.clicked.connect(self.apply_bias)
        layout.addWidget(self.apply_markedness_bias)

        # export the derivation with weights
        self.der_export = QPushButton('Export the cumulative eval with weights')
        self.der_export.clicked.connect(lambda: self.export_derivation(self.table_combined_eval))
        layout.addWidget(self.der_export)

        # Table to display concatenated data
        self.table_combined_eval = QTableWidget()
        layout.addWidget(self.table_combined_eval)

    def apply_bias(self):
        self.run_minimazing_KL(self.combined_cumulative_eval,self.table_combined_eval)
        markedness_constraints = {**agree_dict, **neutral_dict}.keys()
        my_weights = self.optimization.x.tolist()
        all_constraints = dict(zip(self.combined_cumulative_eval.columns.tolist(), my_weights))

        # Updating the dictionary values
        for key in all_constraints:
            if key in markedness_constraints:
                all_constraints[key] = all_constraints[key] + np.float64(self.markedness_bias.value())

        # calculate new harmony and probabilities with markedness bias
        _,solution_with_bias = frequency_and_probabilities(np.array(list(all_constraints.values())), reorder_table(self.combined_cumulative_eval).drop(columns=['output','operation']), update_table=True)
        probabilities = solution_with_bias['probabilities']
        harmonies = solution_with_bias['harmony_score']

        # add the new harmony and probability values
        the_table = self.table_combined_eval
        the_table.insertColumn(2)
        the_table.setHorizontalHeaderItem(2, QTableWidgetItem('probability_with_bias'))

        the_table.insertColumn(4)
        the_table.setHorizontalHeaderItem(4, QTableWidgetItem('harmony_with_bias'))

        for row in range(len(probabilities)):
            my_probability = QTableWidgetItem(str(probabilities[row]))
            my_probability.setFlags(my_probability.flags() & ~Qt.ItemIsEditable)
            the_table.setItem(row, 2, my_probability)

            my_harmony = QTableWidgetItem(str(harmonies[row]))
            my_harmony.setFlags(my_harmony.flags() & ~Qt.ItemIsEditable)
            the_table.setItem(row, 4, my_harmony)

# Main function to run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
