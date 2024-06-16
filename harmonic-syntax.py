# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle
import copy
import csv
import pandas as pd
from typing import List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QMessageBox, QScrollArea, QSizePolicy, QApplication, QDialog
from PyQt5.QtSvg import QSvgWidget
from anytree.exporter import UniqueDotExporter
import sys
from graphviz import Source
import tempfile
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize

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

# Kullback-Leibler divergence 
def KL(p, q): 
    mask = p != 0  # Create a mask to avoid log(0)
    kl_divergence = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    return kl_divergence

# objective function to optimize
def objective_KL(x, df):
    # get weights
    constraint_weights = x[np.newaxis,:]

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

        # calculate divergence
        divergences.append(KL(frequencies, probabilities))
    return np.sum(divergences)

# optimizing function
def weight_optimize(my_tableaux):
    # order the data frame
    columns_to_move = ['input','winner','operation','output']
    remaining_columns = [col for col in my_tableaux.columns if col not in columns_to_move]

    # Create the new column order
    new_order = columns_to_move + remaining_columns
    
    # Reorder the DataFrame
    df_reordered = my_tableaux[new_order]

    # group the cumulative eval in a data frame
    df = df_reordered.drop(columns = ['output', 'operation'])
    
    # number of constraints
    n_constraint = len(remaining_columns)

    # Initial guess for the weights
    initial_weights = np.zeros(n_constraint)
    
    # Define the bounds for the weights (0 to 100)
    bounds = [(0, 100) for _ in range(n_constraint)]

    # Define the objective function that takes only the weights as input
    def objective(weights):
        return objective_KL(weights, df)
    
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
            if current_parent.label and current_parent.label != self.label:
                parent_labels.add(current_parent.label)
            elif current_parent.label == self.label: # if the parent label and the current label are the same
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
    def to_linear(self):
        if not self:
            return ""
        
        if self.name:
            my_name = str(self.name)
        else:
            my_name = ".."

        # add agree features
        agree_feats = [key for key, value in self.agree_feats.items() if value == "1"]
        if agree_feats:
            my_name += " " + ", ".join(agree_feats)

        # add neutral features
        neutral_feats = [key for key, value in self.neutral_feats.items() if value == "1"]
        if neutral_feats:
            my_name += " " + ", ".join(neutral_feats)
                
        result = "[" + my_name
        if self.children:
            result += " " + " ".join(child.to_linear() for child in self.children)
        result += "]"
    
        return result
        
    # merge condition, only checked at the root node (latest operation)
def merge_condition(node):
    violation = 0
    if len(node.children) > 1:
        if node.children[0].merge_feat and node.children[0].merge_feat != node.children[1].label:
            violation += 1

        if node.children[1].merge_feat and node.children[1].merge_feat != node.children[0].label:
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
                empty_agr = None,
                label= str(row['lb'])
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
        if new_left.label == new_right.label:
            new_node.label = new_left.label
        
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
        new_1.label = new_1.children[0].label
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
        new_2.label = new_2.children[1].label
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
    return df

# Define the PyQt5 application and main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the title and size of the main window
        self.setWindowTitle("Harmonic Syntax Tabulator")

        # Create the central widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        # main layout
        mainLayout = QVBoxLayout()

        # left side boxes
        left_side = QVBoxLayout()

        # right side boxes
        right_side = QVBoxLayout()

        # selecting the numeration
        self.label_optimal = QLabel("Double Click on the row name to select it as the optimal output.")
        self.select_numeration = QPushButton('Select Numeration')
        self.select_numeration.clicked.connect(self.import_numeration)

        # export the cumulative eval
        self.eval_export = QPushButton('Export the cumulative eval')
        self.eval_export.setEnabled(False)
        self.eval_export.clicked.connect(self.export_eval)

        # run the weight optimizer
        self.find_weights = QPushButton('Run the weight optimizer')
        self.find_weights.setEnabled(False)
        self.find_weights.clicked.connect(self.run_minimazing_KL)
        
        # displaying the input tree
        self.input_tree = QSvgWidget(self)
        self.input_tree.setFixedSize(500, 700)  # Set fixed size
        self.input_tree.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        left_side.addWidget(self.input_tree)
        left_side.addWidget(self.select_numeration)
        left_side.addWidget(self.label_optimal)
        left_side.addWidget(self.eval_export)
        left_side.addWidget(self.find_weights)

        # displaying eval
        # Create a QTableWidget
        self.table_eval = QTableWidget(self)
        self.table_eval.setColumnCount(len(constraints_dict) + 1)
        self.headers = list(constraints_dict.keys()) # get the keys from constraints dict and available names
        self.table_eval.cellClicked.connect(self.on_cell_clicked) # when the output is clicked, port to tree visualisation
        self.table_eval.horizontalHeader().sectionClicked.connect(self.on_header_clicked) # when the column names are clicked, connect to explanations
        self.table_eval.verticalHeader().sectionDoubleClicked.connect(self.next_cycle)# when the rows are clicked, connect to proceed cycle
        right_side.addWidget(self.table_eval)

        # create empty eval table
        my_columns = list(constraints_dict.keys()) + ['input', 'winner']
        self.cumulative_eval = pd.DataFrame(columns = my_columns)

        # Adding both QVBoxLayouts to a QHBoxLayout
        hLayout = QHBoxLayout()
        hLayout.addLayout(left_side)
        hLayout.addLayout(right_side)

        # Main layout
        mainLayout.addLayout(hLayout)

        # Set the main layout on the central widget
        centralWidget.setLayout(mainLayout)

    def run_minimazing_KL(self):
        self.optimization = weight_optimize(self.cumulative_eval)
        self.label_optimal.setText(f'Was minimzaing function successful? {self.optimization.success}')
        headers = [self.table_eval.horizontalHeaderItem(i).text() for i in range(self.table_eval.columnCount())]
        # Update headers starting from the specified column index
        for i, value in enumerate(self.optimization.x, start=4):
            if i < len(headers):
                headers[i] += f"_({value:.2f})"

        # Set updated headers back to the table
        self.table_eval.setHorizontalHeaderLabels(headers)

        # Resize columns to content
        self.table_eval.resizeColumnsToContents()

    def export_eval(self):
        # Write DataFrame to CSV file
        # order the data frame
        columns_to_move = ['input','winner','operation','output']
        remaining_columns = [col for col in self.cumulative_eval.columns if col not in columns_to_move]

        # Create the new column order
        new_order = columns_to_move + remaining_columns
    
        # Reorder the DataFrame
        self.cumulative_eval = self.cumulative_eval[new_order]
        self.cumulative_eval.to_csv('output.csv', index=False)
        self.label_optimal.setText('Eval exported!')
        self.label_optimal.setStyleSheet('color : green')

    def import_numeration(self):
        # Clear the table
        self.table_eval.clear()
        self.table_eval.setRowCount(0)
        self.table_eval.setColumnCount(0)

        # enable header and output selection
        self.cycle_enabled = True

        # disable weight optimizer
        self.find_weights.setEnabled(False)

        self.numeration_path, _ = QFileDialog.getOpenFileName(self, 'Select Numeration', '.', 'Csv Files (*.csv)')
        self.numeration = read_nodes_csv(self.numeration_path)

        # update headers once for labelling constraints
        self.available_names = [node.name for node in self.numeration]
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
        self.table_eval.clear()
        self.table_eval.setRowCount(0)
        self.table_eval.setColumnCount(0)

        # Set the number of rows in the table
        self.table_eval.setRowCount(len(self.outputs))

        # Set the number of columns in the table
        self.table_eval.setColumnCount(len(self.headers) + 1)
    
        # Set the headers for the table
        headers = ['operation'] + ['output'] + self.headers
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
        self.table_eval.clear()
        self.table_eval.setRowCount(0)
        self.table_eval.setColumnCount(0)

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
            table_widget.setColumnCount(5)
            table_widget.setHorizontalHeaderLabels(['node', 'agr_feats', 'neut_feats', 'dominators', 'used_feats'])

            # Fetch tree structure and node attributes
            root_node = self.outputs[row]
            for pre, _, node in RenderTree(root_node, style=AsciiStyle()):
                row_position = table_widget.rowCount()
                table_widget.insertRow(row_position)
                table_widget.setItem(row_position, 0, QTableWidgetItem(pre + node.name))
                table_widget.setItem(row_position, 1, QTableWidgetItem(str(node.agree_feats)))
                table_widget.setItem(row_position, 2, QTableWidgetItem(str(node.neutral_feats)))
                table_widget.setItem(row_position, 3, QTableWidgetItem(str(node.domination_count())))
                table_widget.setItem(row_position, 4, QTableWidgetItem(str(node.used_feats)))

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

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

