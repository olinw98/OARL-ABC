from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel, QFormLayout, QHBoxLayout, QInputDialog,
    QDialog, QDialogButtonBox, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QTableView, QMenu, QSizePolicy,
    QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QAction
from core.model_trainer import BayesianModel, Parameter, RLTrainer, Model
import json
import numpy as np
import pandas as pd
import logging
import matplotlib

matplotlib.use('QtAgg')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.models = []
        self.input_variable_names = []
        self.imported_input_data = None
        self.imported_output_data = None
        self.epsilon = 0.1

    def initUI(self):
        self.setWindowTitle('Model Trainer')
        self.setGeometry(100, 100, 1000, 700)

        # Create the main layout as a vertical layout
        main_layout = QVBoxLayout()
        column_layout = QHBoxLayout()

        # Left side (Input columns)
        input_layout = QVBoxLayout()
        self.input_columns_label = QLabel("Number of Input Columns (i):", self)
        self.input_columns_input = QLineEdit(self)
        input_layout.addWidget(self.input_columns_label)
        input_layout.addWidget(self.input_columns_input)

        # Right side (Output columns)
        output_layout = QVBoxLayout()
        self.output_columns_label = QLabel("Number of Output Columns (j):", self)
        self.output_columns_input = QLineEdit(self)
        output_layout.addWidget(self.output_columns_label)
        output_layout.addWidget(self.output_columns_input)

        # Add both the input and output layouts to the horizontal layout
        column_layout.addLayout(input_layout)
        column_layout.addLayout(output_layout)
        main_layout.addLayout(column_layout)

        # Add epsilon adjustment
        epsilon_layout = QHBoxLayout()
        epsilon_label = QLabel("Epsilon (exploration rate):", self)
        self.epsilon_spinbox = QDoubleSpinBox(self)
        self.epsilon_spinbox.setRange(0.0, 1.0)
        self.epsilon_spinbox.setDecimals(2)
        self.epsilon_spinbox.setValue(0.1)
        self.epsilon_spinbox.setSingleStep(0.01)

        epsilon_layout.addWidget(epsilon_label)
        epsilon_layout.addWidget(self.epsilon_spinbox)
        main_layout.addLayout(epsilon_layout)

        # Button to add a new model
        add_model_button = QPushButton('Add Model', self)
        add_model_button.clicked.connect(self.open_add_model_window)

        # Button to start training
        start_training_button = QPushButton('Start Training', self)
        start_training_button.clicked.connect(self.start_training)

        # Button to import data
        self.import_data_button = QPushButton('Import Data', self)
        self.import_data_button.clicked.connect(self.import_data)
        main_layout.addWidget(self.import_data_button)

        # Add QLabel to display file path
        self.file_path_label = QLabel("No data imported", self)
        main_layout.addWidget(self.file_path_label)

        # Add Save and Load buttons
        save_models_button = QPushButton('Save Models', self)
        save_models_button.clicked.connect(self.save_models)

        load_models_button = QPushButton('Load Models', self)
        load_models_button.clicked.connect(self.load_models)

        # Horizontal layout for Add Model and Start Training buttons
        add_start_layout = QHBoxLayout()
        add_start_layout.addWidget(add_model_button)
        add_start_layout.addWidget(start_training_button)
        main_layout.addLayout(add_start_layout)

        # Table to display created models
        self.model_table_widget = QTableWidget(self)
        self.model_table_widget.setColumnCount(5)
        self.model_table_widget.setHorizontalHeaderLabels(['Model Name', 'Number of Parameters', 'Function', 'Alpha (α)', 'Beta (β)'])
        self.model_table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.model_table_widget.customContextMenuRequested.connect(self.open_model_context_menu)

        header = self.model_table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        main_layout.addWidget(self.model_table_widget)

        # Label for preview
        preview_label = QLabel('Function Preview:', self)
        main_layout.addWidget(preview_label)

        # Read-only text box for function preview
        self.function_preview = QLineEdit(self)
        self.function_preview.setReadOnly(True)
        main_layout.addWidget(self.function_preview)

        # Horizontal layout for Save and Load buttons
        save_load_layout = QHBoxLayout()
        save_load_layout.addWidget(save_models_button)
        save_load_layout.addWidget(load_models_button)
        main_layout.addLayout(save_load_layout)

        # Set the central widget of the window to use the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)  # Apply the main layout to the central widget
        self.setCentralWidget(central_widget)

    def display_file_path(self, file_path):
        self.file_path_label.setText(f"Data imported from: {file_path}")

    def open_model_context_menu(self, position):
        menu = QMenu()

        delete_action = QAction('Delete Model', self)
        delete_action.triggered.connect(lambda: self.delete_model(self.model_table_widget.rowAt(position.y())))
        menu.addAction(delete_action)

        menu.exec(self.model_table_widget.viewport().mapToGlobal(position))

    def delete_model(self, row):
        if row >= 0:
            model_name = self.models[row].name
            reply = QMessageBox.question(self, 'Delete Model', f"Are you sure you want to delete '{model_name}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                del self.models[row]
                self.update_model_table()

    def open_add_model_window(self):
        logger.info("Opening model creation window.")
        self.add_model_window = AddModelWindow(self)
        self.add_model_window.show()

    def save_models(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Models", "", "JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                models_data = {
                    "version": "1.0",  # Versioning for compatibility
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "models": [
                        {
                            "name": model.name,
                            "alpha": model.alpha,
                            "beta": model.beta,
                            "input_variables": self.input_variable_names,
                            "parameters": [
                                {
                                    "name": param.name,
                                    "shape": param.shape,
                                    "parameters": param.parameters  # Store any parameters like mean, stddev, low, high, etc.
                                }
                                for param in model.parameters
                            ],
                            "function_str": model.function_str  # Save the custom function string (not the lambda itself)
                        }
                        for model in self.models
                    ]
                }

                with open(file_path, 'w') as f:
                    json.dump(models_data, f, indent=4)
                QMessageBox.information(self, "Success", f"Models saved successfully to {file_path}.")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"An error occurred while saving models:\n{e}")
                logger.error(f"Save error: {e}")

    def load_models(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Models", "", "JSON Files (*.json);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    models_data = json.load(f)

                if models_data.get("version") != "1.0":
                    QMessageBox.warning(self, "Version Error", "Incompatible model version.")
                    return

                self.models.clear()

                for model_data in models_data['models']:
                    # Recreate the model
                    model = BayesianModel(
                        name=model_data['name'],
                        alpha=model_data['alpha'],
                        beta=model_data['beta'],
                        fxn=None,  # You can't save lambda functions in JSON, you'll have to reconstruct this manually
                    )

                    # Recreate each Parameter object
                    parameters = [
                        Parameter(
                            name=param_data['name'],
                            shape=param_data['shape'],
                            model=model,
                            **param_data['parameters']  # Use the stored parameters to recreate the Parameter object
                        )
                        for param_data in model_data['parameters']
                    ]

                    model.parameters = parameters

                    if model_data['function_str']:
                        try:
                            # Reconstruct the lambda function using input variables and parameters
                            input_variables = model_data.get('input_variables', [])
                            param_names = [param.name for param in parameters]
                            function_args = input_variables + param_names

                            # Recreate the lambda function from the saved string
                            model.fxn = eval(f"lambda {', '.join(function_args)}: {model_data['function_str']}")
                        except Exception as e:
                            logger.error(f"Failed to recreate the function for model '{model.name}': {e}")

                    # Set the function string and add the model to the list
                    model.function_str = model_data['function_str']  # Load the function string
                    self.models.append(model)

                self.update_model_table()
                QMessageBox.information(self, "Success", f"Models loaded successfully from {file_path}.")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"An error occurred while loading models:\n{e}")
                logger.error(f"Load error: {e}")

    def import_data(self):
        logger.info("Importing data.")
        
        # Get user inputs for i (number of input columns) and j (number of output columns)
        try:
            i = int(self.input_columns_input.text())
            j = int(self.output_columns_input.text())

            if i <= 0 or j <= 0:
                raise ValueError("The number of input and output columns must be positive integers.")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for input and output columns.")
            return

        # File dialog to get the data file
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                # Read the file based on its extension
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    excel_file = pd.ExcelFile(file_path)
                    sheet_names = excel_file.sheet_names
                    sheet_name, ok = QInputDialog.getItem(self, "Select Data", "Available Sheets:", sheet_names, 0, False)

                    if not ok or not sheet_name:
                        return
                    
                    data = pd.read_excel(file_path, sheet_name=sheet_name)

                elif file_path.endswith('.json'):
                    data = pd.read_json(file_path)
                else:
                    QMessageBox.warning(self, "Unsupported File", "Only CSV, Excel, and JSON files are supported.")
                    return
                
                # Check if the provided i and j values are valid
                total_columns = data.shape[1]
                if i + j > total_columns:
                    QMessageBox.warning(self, "Invalid Columns", "The total number of input and output columns exceeds the available columns in the dataset.")
                    return

                # Select input and output columns based on user input
                self.input_variable_names = data.columns[:i].tolist() # Store input variable names
                inputs = data.iloc[:, :i].to_numpy()  # Select input columns
                outputs = data.iloc[:, -j:].to_numpy()  # Select output columns

                self.imported_input_data = inputs  # Store inputs
                self.imported_output_data = outputs  # Store outputs

                # Show a preview of the first 5 rows of data
                self.preview_data(data)

                # Show the file path in the GUI
                self.display_file_path(file_path)

                QMessageBox.information(self, "Data Loaded", f"Data loaded successfully from {file_path}.")

            except Exception as e:
                QMessageBox.warning(self, "Data Import Error", f"An error occurred while loading data:\n{e}")
                logger.error(f"Data import error: {e}")

    def preview_data(self, data):
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle('Data Preview')
        table_view = QTableView(preview_dialog)
        table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(data.columns)

        for index, row in data.head(10).iterrows():
            items = [QStandardItem(str(value)) for value in row]
            model.appendRow(items)

        table_view.setModel(model)
        header = table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(table_view)
        preview_dialog.setLayout(layout)
        preview_dialog.resize(800, 400)
        preview_dialog.exec()

    def start_training(self):
        if not self.models:
            QMessageBox.warning(self, "No Models", "Please add models before starting training.")
            return

        if self.imported_input_data is None or self.imported_output_data is None:
            QMessageBox.warning(self, "No Data", "Please import a dataset before starting training.")
            return

        # Get epsilon value from spinbox
        epsilon = self.epsilon_spinbox.value()

        # Initialize RLTrainer
        trainer = RLTrainer(models=self.models, input_data=self.imported_input_data,
                            output_data=self.imported_output_data, epsilon=epsilon)

        # Run the RLTrainer (this will train synchronously)
        best_model = trainer.run()

        # Update the model table to reflect updated beta distributions
        self.update_model_table()

        # Display a message to the user with the best model name
        QMessageBox.information(self, "Training Complete", f"Training has been successfully completed!\nBest Model: {best_model.name}")

    def update_model_table(self):
        self.model_table_widget.setRowCount(0)

        for model in self.models:
            model.finalize_function_str()

            row_position = self.model_table_widget.rowCount()
            self.model_table_widget.insertRow(row_position)

            # Makes each cell editable
            name_item = QTableWidgetItem(model.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            self.model_table_widget.setItem(row_position, 0, name_item)
            self.model_table_widget.setItem(row_position, 1, QTableWidgetItem(str(len(model.parameters))))
            self.model_table_widget.setItem(row_position, 2, QTableWidgetItem(model.function_str))
            self.model_table_widget.setItem(row_position, 3, QTableWidgetItem(f"{model.alpha:.2f}"))
            self.model_table_widget.setItem(row_position, 4, QTableWidgetItem(f"{model.beta:.2f}"))

        # Connect to edit signal
        self.model_table_widget.itemChanged.connect(self.on_model_item_changed)

    def on_model_item_changed(self, item):
        # Get row and column of the edited item
        row = item.row()
        column = item.column()

        model = self.models[row] # Retrieve the corresponding model object

        if column == 0: # Model name changed
            model.name = item.text()
        elif column == 3: # Alpha changed
            try:
                model.alpha = float(item.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid numeric value for Alpha.")
                return
        elif column == 4: # Beta changed
            try:
                model.beta = float(item.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid numeric value for Beta.")
                return

class AddModelWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.input_variables = parent.input_variable_names
        self.model = None
        self.parameters = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Add New Model')
        self.setGeometry(150, 150, 500, 350)

        form_layout = QFormLayout()

        self.model_name_input = QLineEdit(self)
        self.model_alpha_input = QLineEdit(self)
        self.model_beta_input = QLineEdit(self)
        self.model_function_input = QLineEdit(self)

        form_layout.addRow(QLabel('Model Name:'), self.model_name_input)
        form_layout.addRow(QLabel('Alpha (α):'), self.model_alpha_input)
        form_layout.addRow(QLabel('Beta (β):'), self.model_beta_input)
        form_layout.addRow(QLabel('Custom Function (e.g., "param1 + param2"):'), self.model_function_input)

        add_param_button = QPushButton('Add Parameter', self)
        add_param_button.clicked.connect(self.open_add_param_window)

        self.param_table_widget = QTableWidget(self)
        self.param_table_widget.setColumnCount(3)
        self.param_table_widget.setHorizontalHeaderLabels(['Parameter Name', 'Shape', 'Distribution Parameters'])
        self.param_table_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.param_table_widget.customContextMenuRequested.connect(self.open_param_context_menu)

        param_header = self.param_table_widget.horizontalHeader()
        param_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        param_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.add_model)
        button_box.rejected.connect(self.reject)

        validate_function_button = QPushButton('Validate Function', self)
        validate_function_button.clicked.connect(self.validate_function)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(add_param_button)
        layout.addWidget(self.param_table_widget)
        layout.addWidget(button_box)
        layout.addWidget(validate_function_button)

        self.setLayout(layout)
        self.model_function_input.textChanged.connect(self.update_function_preview)

    def open_param_context_menu(self, position):
        menu = QMenu()

        delete_action = QAction('Delete Parameter', self)
        delete_action.triggered.connect(lambda: self.delete_parameter(self.param_table_widget.rowAt(position.y())))
        menu.addAction(delete_action)

        menu.exec(self.param_table_widget.viewport().mapToGlobal(position))

    def delete_parameter(self, row):
        if row >= 0:
            param_name = self.parameters[row]['name']
            reply = QMessageBox.question(self, 'Delete Parameter', f"Are you sure you want to delete '{param_name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                del self.parameters[row]
                self.update_param_table()

    def open_add_param_window(self):
        self.add_param_window = AddParameterWindow(self)
        self.add_param_window.show()

    def add_model(self):
        model_name = self.model_name_input.text()
        if not model_name:
            QMessageBox.warning(self, "Invalid Input", "Model name cannot be empty.")
            return

        alpha_input = self.model_alpha_input.text()
        beta_input = self.model_beta_input.text()
        alpha = 2.0
        beta = 2.0

        if alpha_input and beta_input:
            try:
                alpha = float(alpha_input)
                beta = float(beta_input)
                if alpha <= 0 or beta <= 0:
                    QMessageBox.warning(self, "Invalid Alpha/Beta", "Alpha and Beta must be positive numbers.")
                    return
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for Alpha and Beta.")
                return
        elif alpha_input or beta_input:
            QMessageBox.warning(self, "Incomplete Input", "Both Alpha and Beta fields need to be filled for custom Beta distribution values.")
            return

        custom_function = None
        function_input = self.model_function_input.text()

        if function_input:
            try:
                param_names = [param['name'] for param in self.parameters]
                function_args = self.parent.input_variable_names + param_names

                function_code = f"lambda {', '.join(function_args)}: {function_input}"
                custom_function = eval(function_code)
            except Exception as e:
                QMessageBox.warning(self, "Invalid Function", f"Invalid custom function provided: {e}")
                return

        function_str = function_input or "Default Function"
        self.model = BayesianModel(name=model_name, alpha=alpha, beta=beta, fxn=custom_function, function_str=function_str)

        for param_details in self.parameters:
            Parameter(name=param_details['name'], model=self.model, shape=param_details['shape'], **param_details['params'])

        self.parent.models.append(self.model)
        self.parent.update_model_table()

        self.accept()

    def validate_function(self):
        function_input = self.model_function_input.text()
        if not function_input.strip():
            QMessageBox.warning(self, "Empty Function", "Please enter a custom function to validate (e.g., 'param1 + X1').")
            return
        
        try:
            param_names = [param['name'] for param in self.parameters]
            function_args = self.input_variables + param_names

            function_code = f"lambda {', '.join(function_args)}: {function_input}"
            compile(function_code, '<string>', 'eval')

            QMessageBox.information(self, "Valid Function", "The custom function is valid")
        except Exception as e:
            QMessageBox.warning(self, "Invalid Function", f"The custom function is invalid:\n{e}")

    def update_param_table(self):
        self.param_table_widget.setRowCount(0)

        for param in self.parameters:
            row_position = self.param_table_widget.rowCount()
            self.param_table_widget.insertRow(row_position)

            # Make each parameter name editable
            name_item = QTableWidgetItem(param['name'])
            name_item.setFlags(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            self.param_table_widget.setItem(row_position, 0, name_item)
            self.param_table_widget.setItem(row_position, 1, QTableWidgetItem(param['shape']))
            param_info = ', '.join(f"{k}={v}" for k, v in param['params'].items())
            self.param_table_widget.setItem(row_position, 2, QTableWidgetItem(param_info))

        # Connect to edit signal
        self.param_table_widget.itemChanged.connect(self.on_param_item_changed)

    def on_param_item_changed(self, item):
        row = item.row()
        column = item.column()

        param = self.parameters[row] # Retrieve the corresponding parameter object

        if column == 0: # Parameter name changed
            param['name'] = item.text()

    def update_function_preview(self):
        function_input = self.model_function_input.text()
        param_names = [param['name'] for param in self.parameters]
        if function_input and param_names:
            function_code = f"lambda {', '.join(self.input_variables + param_names)}: {function_input}"
            self.parent.function_preview.setText(function_code)
        else:
            self.parent.function_preview.setText("Add parameters to define function.")

class AddParameterWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Add New Parameter')
        self.setGeometry(200, 200, 300, 250)

        self.form_layout = QFormLayout()

        self.param_name_input = QLineEdit(self)
        self.param_shape_input = QComboBox(self)
        self.param_shape_input.addItems(['normal', 'uniform', 'beta', 'triangular'])
        self.param_shape_input.currentTextChanged.connect(self.update_param_inputs)

        self.form_layout.addRow(QLabel('Parameter Name:'), self.param_name_input)
        self.form_layout.addRow(QLabel('Parameter Shape:'), self.param_shape_input)

        self.param_inputs = {}
        self.update_param_inputs()

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        button_box.accepted.connect(self.add_parameter)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(self.form_layout)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def update_param_inputs(self):
        for i in reversed(range(self.form_layout.rowCount())):
            widget = self.form_layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()
            if widget and widget not in [self.param_name_input, self.param_shape_input]:
                self.form_layout.removeRow(i)

        selected_shape = self.param_shape_input.currentText()

        if selected_shape == 'normal':
            self.param_inputs = {
                'mean': QLineEdit(self),
                'stddev': QLineEdit(self)
            }
            self.form_layout.addRow(QLabel('Mean:'), self.param_inputs['mean'])
            self.form_layout.addRow(QLabel('Stddev:'), self.param_inputs['stddev'])
        elif selected_shape == 'uniform':
            self.param_inputs = {
                'low': QLineEdit(self),
                'high': QLineEdit(self)
            }
            self.form_layout.addRow(QLabel('Low:'), self.param_inputs['low'])
            self.form_layout.addRow(QLabel('High:'), self.param_inputs['high'])
        elif selected_shape == 'beta':
            self.param_inputs = {
                'alpha': QLineEdit(self),
                'beta': QLineEdit(self)
            }
            self.form_layout.addRow(QLabel('Alpha:'), self.param_inputs['alpha'])
            self.form_layout.addRow(QLabel('Beta:'), self.param_inputs['beta'])
        elif selected_shape == 'triangular':
            self.param_inputs = {
                'low': QLineEdit(self),
                'mode': QLineEdit(self),
                'high': QLineEdit(self)
            }
            self.form_layout.addRow(QLabel('Low:'), self.param_inputs['low'])
            self.form_layout.addRow(QLabel('Mode:'), self.param_inputs['mode'])
            self.form_layout.addRow(QLabel('High:'), self.param_inputs['high'])

    def add_parameter(self):
        param_name = self.param_name_input.text()
        param_shape = self.param_shape_input.currentText()

        params = {}
        try:
            if param_shape == 'normal':
                params['mean'] = float(self.param_inputs['mean'].text())
                params['stddev'] = float(self.param_inputs['stddev'].text())
            elif param_shape == 'uniform':
                params['low'] = float(self.param_inputs['low'].text())
                params['high'] = float(self.param_inputs['high'].text())
            elif param_shape == 'beta':
                params['alpha'] = float(self.param_inputs['alpha'].text())
                params['beta'] = float(self.param_inputs['beta'].text())
            elif param_shape == 'triangular':
                params['low'] = float(self.param_inputs['low'].text())
                params['mode'] = float(self.param_inputs['mode'].text())
                params['high'] = float(self.param_inputs['high'].text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for the parameter fields.")
            return

        if param_shape == 'uniform' and params['low'] >= params['high']:
            QMessageBox.warning(self, "Invalid Parameters", "'Low' must be less than 'High' for uniform distribution.")
            return
        elif param_shape == 'beta' and (params['alpha'] <= 0 or params['beta'] <= 0):
            QMessageBox.warning(self, "Invalid Parameters", "For beta distribution, 'Alpha' and 'Beta' must be positive.")
            return
        elif param_shape == 'triangular' and not (params['low'] <= params['mode'] <= params['high']):
            QMessageBox.warning(self, "Invalid Parameters", "For triangular distribution, 'Low' ≤ 'Mode' ≤ 'High'.")
            return

        if any(param['name'].lower() == param_name.lower() for param in self.parent.parameters):
            QMessageBox.warning(self, "Duplicate Parameter", f"A parameter with the name '{param_name}' already exists. Please choose a different name.")
            return

        if param_name and all(params.values() or value == 0 for value in params.values()):
            param_details = {
                'name': param_name,
                'shape': param_shape,
                'params': params
            }
            self.parent.parameters.append(param_details)
            self.parent.update_param_table()
            self.accept()
        else:
            QMessageBox.warning(self, "Invalid Input", "All parameter fields must be filled.")

if __name__ == '__main__':
    app = QApplication([])
    trainer_app = ModelTrainerApp()
    trainer_app.show()
    app.exec()