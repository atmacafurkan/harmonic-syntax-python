# Using anytree in Python
import copy
import pandas as pd
from typing import List
from anytree.exporter import UniqueDotExporter
from graphviz import Source
import tempfile
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from anytree import Node

def remove_keys(dictionary: dict, specific_keys: list, suffix: str):
    new_dict = {}
    for key, value in dictionary.items():
        if key in specific_keys or key.endswith(suffix):
            continue
        new_dict[key] = value
    return new_dict
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
    def __init__(self, name, label: str = None, merge_feat: str = None, neutral_feats: dict = None, empty_agr: dict = None, result_feats: dict = None, agree_feats: dict = None, used_feats: dict = None, other_nodes= None, operation = None, exhaust_ws = None, parent = None, children = None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else None
        self.merge_feat = merge_feat if merge_feat is not None and merge_feat != '' else None
        self.agree_feats = agree_feats if agree_feats is not None else {}
        self.neutral_feats = neutral_feats if neutral_feats is not None else {}
        self.used_feats = used_feats if used_feats is not None else {key: 0 for key in self.neutral_feats.keys()}
        self.empty_agr = empty_agr if empty_agr is not None else {} 
        self.result_feats = result_feats if result_feats is not None else {'merge_cond' : 0, 'exhaust_ws': 0, 'label_cons': 0, **self.agree_feats, **self.neutral_feats, **self.empty_agr}
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

    def evaluate_constraints(self, encountered_nodes: set = None, initial_eval: dict = None):
        # Initialize a result dictionary for both agree_feats and neutral_feats
        result_feats = {} if initial_eval is None else initial_eval.copy()

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
            if key not in result_feats.keys():
                result_feats[key] = 0
            result_feats[key] += int(value) * domination_count 
        
        # Recursive call for each child node
        for child in self.children:
            result_feats = child.evaluate_constraints(encountered_nodes, result_feats)

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
        agree_feats = [key for key, value in self.agree_feats.items() if value == 1]
        # add neutral features
        neutral_feats = [key for key, value in self.neutral_feats.items() if value == 1]

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
        if not node.children[0].name or not node.children[1].name:
            violation +=1
            return violation

        if (node.children[0].merge_feat and node.children[0].merge_feat != node.children[1].name):
            violation += 1

        if (node.children[1].merge_feat and node.children[1].merge_feat != node.children[0].name):
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

def parse_feats_old(feat_str):
    empty_dict = {} # for some f*cked up reason return {} throws an indentation error
    if feat_str is not None and feat_str.strip():
        return dict(pair.split(':') for pair in feat_str.split('-'))
    else: 
        return empty_dict
    
def parse_feats(feat_str):
    empty_dict = {}  # Placeholder empty dictionary
    if feat_str is not None and feat_str.strip():
        return {key: int(value) for key, value in (pair.split(':') for pair in feat_str.split('-'))}
    else:
        return empty_dict
    
def replace_suffix(dictionary, replace_me, replace_with):
    new_dict = {}
    for key in dictionary.keys():
        if key.endswith(replace_me):
            new_key = key[:-len(replace_me)] + replace_with
            new_dict[new_key] = dictionary[key]
        else:
            new_dict[key] = dictionary[key]
    return new_dict

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
            if value == 1 and new_1.children[0].used_feats[key] == 1:
                new_1_neutral[key] = 0  # Reset to 0

        new_1.neutral_feats = new_1_neutral
        new_1.used_feats = new_1.children[0].used_feats
        new_1.operation = "Label"
        new_1.exhaust_ws = 0
        my_nodes.append(new_1)

        # take from right
        new_2 = clone_tree(my_node)
        new_2.merge_feat = {}
        new_2.name = new_2.children[1].name
        new_2.agree_feats = new_2.children[1].agree_feats

        # check agreement
        new_2_neutral = new_2.children[1].neutral_feats.copy()  # Make a copy to modify
        for key, value in new_2_neutral.items():
            if value == 1 and new_2.children[1].used_feats[key] == 1:
                new_2_neutral[key] = 0  # Reset to 0

        new_2.used_feats = new_2.children[1].used_feats
        new_2.neutral_feats = new_2_neutral
        new_2.operation = "Label"
        new_2.exhaust_ws = 0
        my_nodes.append(new_2)
    return(my_nodes)

# # empty agreement in a combination of the agreement features
# def Agree(my_node):
#     new_list = []    
#     # empty agreement if it is a labelled node and has agreement features
#     if len(my_node.name) > 0 and "1" in my_node.agree_feats.values():
#         # Create a copy to avoid modifying the original node's attributes
#         my_agr = my_node.agree_feats.copy()
#         my_empty = my_node.empty_agr.copy()

#         # Find all keys with value '1'
#         keys_with_one = [key for key, value in my_agr.items() if value == '1']
        
#         # Generate all non-empty combinations of these keys
#         for r in range(1, len(keys_with_one) + 1):
#             for combination in itertools.combinations(keys_with_one, r):
#                 new_node = clone_tree(my_node)
                
#                 # Create copies for modification
#                 temp_agr = my_agr.copy()
#                 temp_empty = my_empty.copy()
                
#                 # Update the copied dictionaries based on the current combination
#                 for key in combination:
#                     temp_agr[key] = 0
#                     temp_empty[key.replace('_agr', '') + '_mt'] = 1
                
#                 # Set the new attributes and other properties
#                 new_node.agree_feats = temp_agr
#                 new_node.empty_agr = temp_empty
#                 new_node.operation = "mtAgree"
#                 new_node.exhaust_ws = 0

#                 # Add the new node to the list if there are changes
#                 if my_node.agree_feats != new_node.agree_feats:
#                     new_list.append(new_node)

# Agree function, only under sisterhood
def Agree(my_node):
    new_list = []    
    # empty agreement if it is a labelled node and has agreement features
    if 1 in my_node.agree_feats.values():
        new_node = clone_tree(my_node)

        my_agr = my_node.agree_feats.copy()  # Create a copy to avoid modifying the original node's attributes
        my_empty = my_node.empty_agr.copy()
        for key, value in my_agr.items():
            if my_agr[key] == 1:
                my_agr[key] = 0
                my_empty[key.replace('_agr', '') + '_mt'] = 1
        
        new_node.agree_feats = my_agr
        new_node.empty_agr = my_empty
        new_node.operation = "mtAgree"
        new_node.exhaust_ws = 0
        if my_node.agree_feats != new_node.agree_feats:
            new_list.append(new_node)    
        return new_list
                  
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
                my_left_agr[key + "_agr"] = 0
                if value == 1:
                    my_used_right[key] = 1
                
        newer_node.children[0].agree_feats = my_left_agr
        newer_node.children[1].used_feats = my_used_right

        # Agree right
        my_right_agr = my_node.children[1].agree_feats
        my_left_feats = my_node.children[0].neutral_feats
        my_used_left = my_node.children[0].used_feats

        for key, value in my_left_feats.items():
            if key + "_agr" in my_right_agr and value == my_right_agr[key + "_agr"]:
                my_right_agr[key + "_agr"] = 0
                if value == 1:
                    my_used_left[key] = 1

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

