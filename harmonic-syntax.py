# Using anytree in Python
from anytree import Node, RenderTree, AsciiStyle
import copy
import csv
from typing import List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QMessageBox, QScrollArea, QSizePolicy, QApplication, QDialog
from PyQt5.QtSvg import QSvgWidget
from anytree.exporter import UniqueDotExporter
import sys
from graphviz import Source
import tempfile

agree_dict = {'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0}
neutral_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
empty_dict = {'case_mt': 0, 'wh_mt': 0, 'foc_mt': 0}
used_feats_dict = {'case': 0, 'wh' : 0, 'foc' : 0}
constraints_dict = {'merge_cond': 0, 'exhaust_ws': 0,'label_cons': 0, 'case_agr': 0, 'wh_agr' : 0, 'foc_agr' : 0, 'case': 0, 'wh' : 0, 'foc' : 0, 'case_mt' : 0, 'wh_mt': 0, 'foc_mt': 0}
explanations_dict = {'operation': 'The operation name given for easy intrepretation. xMerge, iMerge, and rMerge are all one operation Merge.',
                     'output': 'A linear representation of the output. You can click on it to view the visual representation.',
                     'merge_cond': 'Merge condition constraint, a constraint tied to the operation Merge. It is violated when the merge feature of one item does not match the label of the other.',
                     'label_cons': 'Labelling constraint, a constraint tied to the operation Merge. It is violated when the result of the merge does not have a label.',
                     'case_agr': 'Markedness constraint for case agreement feature.',
                     'wh_agr': 'Markedness constraint for wh agreement feature.',
                     'foc_agr': 'Markedness constraint for focus agreement feature.',
                     'case': 'Markedness constraint for case feature.',
                     'wh': 'Markedness constraint for wh feature.',
                     'foc': 'Markedness constraint for focus feature.',
                     'case_mt': 'Empty agreement constraint for case, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.',
                     'foc_mt': 'Empty agreement constraint for focus, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.',
                     'wh_mt': 'Empty agreement constraint for wh, a constraint tied to the operation Agree. It is violated when the agreement features of the root node is satisfied unilaterally.'
                     }

# Custom node class with a named list field
class SyntaxNode(Node):
    def __init__(self, name, label = None, merge_feat = None, neutral_feats = None, empty_agr = None, result_feats = None, agree_feats = None, other_nodes = None, operation = None, exhaust_ws = None, parent = None, children = None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else None
        self.merge_feat = merge_feat if merge_feat is not None and merge_feat != '' else None
        self.agree_feats = agree_feats if agree_feats is not None else agree_dict
        self.neutral_feats = neutral_feats if neutral_feats is not None else neutral_dict
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

# for labelling constraint, only checked at the root node (latest operation)
def label_constraint(node):
    # if the node does not have a name and the operation is Merge, return 1
    if not node.name and node.operation in ["xMerge","iMerge"]:
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
    
    cloned_node = SyntaxNode(
        name=node.name,
        label=node.label,
        merge_feat = node.merge_feat,
        agree_feats = node.agree_feats,
        empty_agr=node.empty_agr,
        neutral_feats=node.neutral_feats,
        exhaust_ws=node.exhaust_ws,
        operation=node.operation,
        result_feats=node.result_feats,
        other_nodes=node.other_nodes
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
    #reflexive_merge = copy.deepcopy(cloned_nodes[1])
    #reflexive_merge.operation = "rMerge"
    #reflexive_merge.exhaust_ws = 1
    #output_nodes.append(reflexive_merge)

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
        new_1.operation = "Label"
        new_1.exhaust_ws = 0
        my_nodes.append(new_1)

        # take from right
        new_2 = clone_tree(my_node)
        new_2.other_nodes = my_node.other_nodes
        new_2.merge_feat = {}
        new_2.name = new_2.children[1].name
        new_2.label = new_2.children[1].label
        new_2.agree_feats = new_2.children[1].agree_feats
        new_2.neutral_feats = new_2.children[1].neutral_feats
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
        new_node.other_nodes = my_node.other_nodes

        my_agr = my_node.agree_feats.copy()  # Create a copy to avoid modifying the original node's attributes
        my_empty = my_node.empty_agr.copy()
        for key, value in my_agr.items():
            if my_agr[key] == '1':
                my_agr[key] = 0
                my_empty[key.replace('_agr', '') + '_mt'] = 1
        
        new_node.agree_feats = my_agr
        new_node.empty_agr = my_empty
        new_node.operation = "Agree"
        new_node.exhaust_ws = 0
        if my_node.agree_feats != new_node.agree_feats:
            new_list.append(new_node)
                  
    if len(my_node.children) > 0 and len(my_node.name) == 0:
        # Create a new node
        newer_node = clone_tree(my_node)
        newer_node.other_nodes = my_node.other_nodes

        # Agree left
        my_left_agr = my_node.children[0].agree_feats
        my_right_feats = my_node.children[1].neutral_feats
        for key, value in my_right_feats.items():
            if key + "_agr" in my_left_agr and value == my_left_agr[key + "_agr"]:
                my_left_agr[key + "_agr"] = 0

        newer_node.children[0].agree_feats = my_left_agr

        # Agree right
        my_right_agr = my_node.children[1].agree_feats
        my_left_feats = my_node.children[0].neutral_feats
        for key, value in my_left_feats.items():
            if key + "_agr" in my_right_agr and value == my_right_agr[key + "_agr"]:
                my_right_agr[key + "_agr"] = 0

        # Only update the right child's agree_feats
        newer_node.children[1].agree_feats = my_right_agr
        newer_node.operation = "Agree children"
        newer_node.exhaust_ws = 0

        if newer_node.children[0].agree_feats != my_node.children[0].agree_feats:
            new_list.append(newer_node) #if there has been a change in agree_feats values, append the new_node to the list
    return new_list

# function to form outputs from an input
def proceed_cycle(input_node):
    output_nodes = []
     
    for_merge = copy.deepcopy(input_node)
    output_nodes.extend(Merge(for_merge)) # carry out merge

    for_label = copy.deepcopy(input_node)
    output_nodes.extend(Label(for_label)) # carry out label

    for_agree = copy.deepcopy(input_node)
    output_nodes.extend(Agree(for_agree)) # carry out agree

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
        
        # displaying the input tree
        self.input_tree = QSvgWidget(self)

        left_side.addWidget(self.input_tree)
        left_side.addWidget(self.select_numeration)
        left_side.addWidget(self.label_optimal)

        # displaying eval
        # Create a QTableWidget
        self.table_eval = QTableWidget(self)
        self.table_eval.setColumnCount(len(constraints_dict) + 1)
        self.headers = list(constraints_dict.keys())
        self.table_eval.cellClicked.connect(self.on_cell_clicked) # when the output is clicked, port to tree visualisation
        self.table_eval.horizontalHeader().sectionClicked.connect(self.on_header_clicked) # when the column names are clicked, connect to explanations
        self.table_eval.verticalHeader().sectionDoubleClicked.connect(self.next_cycle)# when the rows are clicked, connect to proceed cycle
        right_side.addWidget(self.table_eval)

        # Adding both QVBoxLayouts to a QHBoxLayout
        hLayout = QHBoxLayout()
        hLayout.addLayout(left_side)
        hLayout.addLayout(right_side)

        # Main layout
        mainLayout.addLayout(hLayout)

        # Set the main layout on the central widget
        centralWidget.setLayout(mainLayout)

    def import_numeration(self):
        self.numeration_path, _ = QFileDialog.getOpenFileName(self, 'Select Numeration', '.', 'Csv Files (*.csv)')
        self.numeration = read_nodes_csv(self.numeration_path)
        if self.numeration_path:
            # update the input tree
            my_input = generate_svg_content(self.numeration[0])

            # Load the SVG into QSvgWidget
            self.input_tree.load(my_input.encode('utf-8'))

            # run the first cycle for the input
            self.outputs = proceed_cycle(self.numeration[0])

            # Update the evaluation
            self.update_eval()

    def update_eval(self):
        # Set the number of rows in the table
        self.table_eval.setRowCount(len(self.outputs))

        # Set the number of columns in the table
        self.table_eval.setColumnCount(len(self.headers) + 1)
    
            # Set the headers for the table
        headers = ['operation'] + ['output'] + self.headers
        self.table_eval.setHorizontalHeaderLabels(headers)
    
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
            data_dict['exhaust_ws'] = new_node.exhaust_ws
            data_dict = empty_agreement(new_node, data_dict) # check for empty agreement
            for col, key in enumerate(self.headers, start=2):  # Start from column 1 to skip the first column
                value = data_dict.get(key, '')
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table_eval.setItem(row, col, item)
    
        # Resize columns to content
        self.table_eval.resizeColumnsToContents()

    def next_cycle(self, logicalIndex):
        # get the selected output
        selected_output = self.outputs[logicalIndex]
        
        # produce the new outputs
        self.outputs = proceed_cycle(selected_output)

        # update the input tree
        my_input = generate_svg_content(selected_output)
        # Load the SVG into QSvgWidget
        self.input_tree.load(my_input.encode('utf-8'))

        # update the evaluation table
        self.update_eval()

    def on_cell_clicked(self, row, column):
        # Custom function to be executed when a cell is clicked
        if column == 1:
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
            table_widget.setColumnCount(4)
            table_widget.setHorizontalHeaderLabels(['node', 'agr_feats', 'neut_feats', 'dominators'])

            # Fetch tree structure and node attributes
            root_node = self.outputs[row]
            for pre, _, node in RenderTree(root_node, style=AsciiStyle()):
                row_position = table_widget.rowCount()
                table_widget.insertRow(row_position)
                table_widget.setItem(row_position, 0, QTableWidgetItem(pre + node.name))
                table_widget.setItem(row_position, 1, QTableWidgetItem(str(node.agree_feats)))
                table_widget.setItem(row_position, 2, QTableWidgetItem(str(node.neutral_feats)))
                table_widget.setItem(row_position, 3, QTableWidgetItem(str(node.domination_count())))

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

