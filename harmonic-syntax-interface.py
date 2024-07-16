from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QMessageBox, QScrollArea, QSizePolicy, QApplication, QDialog, QTabWidget, QDoubleSpinBox
from PyQt5.QtSvg import QSvgWidget
import csv
import sys
import os
from tabulate import tabulate
from pylatexenc.latexencode import unicode_to_latex
from anytree import RenderTree, AsciiStyle
import re
from harmonicSyntax import *

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

    # function to read the csv file as nodes
    def read_nodes_csv(self, csv_file_path: str) -> List[SyntaxNode]:
        nodes = []
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                my_agree = parse_feats(row['ac'])
                my_neutral = parse_feats(row['ft'])
                my_empty = {key: 0 for key in replace_suffix(parse_feats(row['ac']),"agr","mt").keys()}
                node = SyntaxNode(
                    name = row['it'],
                    merge_feat = str(row['mc']),
                    agree_feats = my_agree,
                    neutral_feats = my_neutral,
                    empty_agr = my_empty,
                    result_feats =  {'merge_cond' : 0, 'exhaust_ws': 0, 'label_cons': 0, **my_agree, **my_neutral, **my_empty}
                    )
                nodes.append(node)

        # get the list of constraints for this derivation
        self.constraints_set = set()
        for node in nodes:
            attributes = node.result_feats
            for key in attributes.keys():
                self.constraints_set.add(key)

        # make the explanations for the list of constraints
        self.explanations_dict = {'input':'The input for the derivation cycle',
                     'winner':'The optimal output of the derivation cycle',
                     'operation': 'The operation name given for easy intrepretation. xMerge, iMerge, and rMerge are all one operation Merge.',
                     'output': 'A linear representation of the output. You can click on it to view the visual representation.',
                     'merge_cond': 'Merge condition constraint, a constraint tied to the operation Merge. It is violated when the merge feature of one item does not match the label of the other.',
                     'exhaust_ws':'Exhuast Workspace, this is violated when both the items used by merge are in the input.',
                     'label_cons': 'Labelling constraint, a constraint tied to the operation Merge. It is violated when the result of the merge does not have a label.'
                     }
        
        # Update the other_nodes field for each node by cloning the other nodes
        for i, node in enumerate(nodes):
            node.other_nodes = [clone_tree(other_node) for j, other_node in enumerate(nodes) if i != j]

        return nodes

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
        self.table_eval.cellClicked.connect(self.on_cell_clicked) # when the output is clicked, port to tree visualisation
        #self.table_eval.horizontalHeader().sectionClicked.connect(self.on_header_clicked) # when the column names are clicked, connect to explanations
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

        # enable header and output selection
        self.cycle_enabled = True

        # disable weight optimizer
        self.find_weights.setEnabled(False)

        # disable derivation export
        self.der_export.setEnabled(False)

        self.numeration_path, _ = QFileDialog.getOpenFileName(self, 'Select Numeration', '.', 'Csv Files (*.csv)')
        self.numeration = self.read_nodes_csv(self.numeration_path)

        # create empty eval table
        my_columns = list(self.constraints_set) + ['input', 'winner']

        self.cumulative_eval = pd.DataFrame(columns = my_columns)

        # update headers once for labelling constraints
        self.available_names = None
        self.available_names = [node.name for node in self.numeration]
        
        self.headers = list(self.constraints_set) # get the keys from constraints dict and available names
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
        headers = ['operation'] + ['output'] + [str(header) for header in self.headers]

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
            message = f"{self.explanations_dict[header_label]}"
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
        dx = df.drop(columns=['input','derivation','harmony','harmonyBias'], errors = 'ignore')
        dx = dx.loc[:, (dx != 0).any(axis=0)]

        # rename columns for reducing size
        latex_names = {'merge_cond': 'mc', 'exhaust_ws': 'xws', 'label_cons': 'lab', 'operation': 'opr.', 'winner': 'W', 'probability': 'prb.', 'LB' : 'lb', 'harmony' : 'H'}

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
        table = re.sub(r'begin\{tabular\}', 'begin{tabular}{\\\\linewidth}', table)
        table = re.sub(r'rll', 'rlX', table)
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

        # replace operation names
        table = table.replace("Label","Lbl")
        table = table.replace("xMerge","xMrg")
        table = table.replace("iMerge","iMrg")
        table = table.replace("rMerge","rMrg")
        table = table.replace("Agree","Agr")

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
        markedness_constraints = remove_keys({key:0 for key in self.constraints_set}, ["merge_cond","exhaust_ws", "label_cons"], "_mt") # remove GEN constraints 
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
        the_table.setHorizontalHeaderItem(2, QTableWidgetItem('probabilityBias'))

        the_table.insertColumn(4)
        the_table.setHorizontalHeaderItem(4, QTableWidgetItem('harmonyBias'))

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