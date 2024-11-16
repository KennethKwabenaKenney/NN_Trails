import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import seaborn as sns
#%% FFNN with BP Class
class NeuralNetwork:
    def __init__(self, inputLayerSize, outputLayerSize):
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSizes = []
        self.weights = []
        self.biases = []
        self.activation = lambda x: 1 / (1 + np.exp(-x))
        self.activation_derivative = lambda x: x * (1 - x)

    def add_hidden_layer(self, layer_size):
        self.hiddenLayerSizes.append(layer_size)

    def initialize_weights(self):
        layer_sizes = [self.inputLayerSize] + self.hiddenLayerSizes + [self.outputLayerSize]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(1, layer_sizes[i + 1]))

    def forward(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            self.layer_outputs.append(self.activation(np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]))
        return self.layer_outputs[-1]

    def backward(self, X, y, learning_rate):
        deltas = [None] * len(self.weights)
        # Calculate output layer delta
        error = y - self.layer_outputs[-1]
        deltas[-1] = error * self.activation_derivative(self.layer_outputs[-1])
        # Backpropagate the deltas
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = deltas[i + 1].dot(self.weights[i + 1].T) * self.activation_derivative(self.layer_outputs[i + 1])
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.layer_outputs[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, X_train, Y_train, epochs, learning_rate):
        self.initialize_weights()
        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, Y_train, learning_rate)

    def predict(self, X_test):
        return self.forward(X_test)

#%% Support Definitions
app = QApplication([])  # Create a PyQt application
def select_file(root_dir, title="Select a file", file_filter="CSV files (*.csv)"):
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle(title)
    file_dialog.setFileMode(QFileDialog.ExistingFile) 
    file_dialog.setNameFilter(file_filter)
    file_dialog.setDirectory(root_dir)  # Set the root directory
    if file_dialog.exec_():
        file_paths = file_dialog.selectedFiles()
        return file_paths[0]
    return None

def reclassify_labels(X_train, y_train, X_test, y_test):
    # Specify the reclassification label
    reclassification_label = int(input("Enter the reclassification label: "))  # Convert input to integer

    # Ensure y_train and y_test are pandas Series
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)
    removed_values_train = unique_labels_train - common_labels
    removed_values_test = unique_labels_test - common_labels
   
    print("Reclassified non-common values from y_train:")
    for label in removed_values_train:
        count = sum(y_train == label)
        print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

    print("\nReclassified non-common values from y_test:")
    for label in removed_values_test:
        count = sum(y_test == label)
        print(f"{label} reclassified to {reclassification_label}: {count} occurrences")

    # Reclassify labels not present in both y_train and y_test
    y_train = y_train.replace(list(removed_values_train), reclassification_label)
    y_test = y_test.replace(list(removed_values_test), reclassification_label)
    
    # Update common_labels after reclassification
    common_labels = common_labels.union([reclassification_label])

    return X_train, y_train, X_test, y_test

def synchronize_labels(X_train, y_train, X_test, y_test, excel_file_path):
    # Ensure labels in y_test are present in y_train, and vice versa
    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)

    # Find and print removed values
    removed_values_train = set(y_train) - common_labels
    removed_values_test = set(y_test) - common_labels
    
    # Load Excel file
    excel_data = pd.read_excel(excel_file_path)
    ext_class_labels_mapping = dict(zip(excel_data['Ext_Class'], excel_data['Labels']))
    
    print("Removed values from y_train:")
    for label in removed_values_train:
        count = sum(y_train == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    print("\nRemoved values from y_test:")
    for label in removed_values_test:
        count = sum(y_test == label)
        ext_class_label = ext_class_labels_mapping.get(label, 'Not Found')
        print(f"{label} ({ext_class_label}): {count} occurrences")

    # Remove rows with labels not present in both y_train and y_test
    mask_train = y_train.isin(common_labels)
    mask_test = y_test.isin(common_labels)

    X_train_synchronized = X_train[mask_train]
    y_train_synchronized = y_train[mask_train]

    X_test_synchronized = X_test[mask_test]
    y_test_synchronized = y_test[mask_test]

    return X_train_synchronized, y_train_synchronized, X_test_synchronized, y_test_synchronized

#%% Implementation
# Labels file path
labels_file_path = r'D:\ODOT_SPR866\My Label Data Work\Sample Label data for testing\Ext_Class_labels.xlsx'

# Parent file path for datasets
root_dir = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\5_ind_objects'

# Select the training dataset
training_file = select_file(root_dir,"Select Training Data")
print("Selected training dataset:", training_file)

# Select the testing dataset
testing_file = select_file(root_dir, "Select Testing Data")
print("Selected testing dataset:", testing_file)

# Check selected files selected and load CSV
if training_file and testing_file:
    # Load the selected CSV files into DataFrames
    Train_data = pd.read_csv(training_file)
    Test_data = pd.read_csv(testing_file)

    print("Training Data Loaded. Shape:", Train_data.shape)
    print("Testing Data Loaded. Shape:", Test_data.shape)
else:
    print("No file selected. Please select a valid CSV file.")

# Training
Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class']   #Columns to exclude from the X data
X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)    
y_train = Train_data['Ext_class']

# Testing
Test_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class']   #Columns to exclude from the X data
X_test = Test_data.drop(columns=Test_cols_to_remove, axis=1)    
y_test = Test_data['Ext_class']

# Synchronize or reclassify labels
print("Select an option:\n1. Reclassify non-common labels\n2. Synchronize labels")
action = input("Enter 1 or 2: ").strip()
if action == '1':
        X_train, y_train, X_test, y_test = reclassify_labels(X_train, y_train, X_test, y_test)
elif action == '2':
    X_train, y_train, X_test, y_test = synchronize_labels(X_train, y_train, X_test, y_test, labels_file_path)
else:
    print("Invalid option entered. No changes made.")

#%% Example usage:
# # Assuming X_train, Y_train, X_test, Y_test are loaded from CSV
# X_train = np.random.rand(100, 10)  # Example random data for X_train
# X_test = np.random.rand(50, 10)    # Example random data for X_test

# # Generate random labels for demonstration (replace with your actual labels)
# Y_train = np.random.choice([99, 110, 250, 500], size=(100,))
# Y_test = np.random.choice([99, 110, 250, 500], size=(50,))


inputLayerSize = X_train.shape[1]
outputLayerSize = len(np.unique(y_train))

# One-hot encode Y_train and Y_test
y_train_encoded = pd.get_dummies(y_train).values
y_test_encoded = pd.get_dummies(y_test).values

# User inputs
hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_layer = []
for i in range(hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {i+1}: "))
    neurons_per_layer.append(neurons)

epochs = int(input("Enter the number of epochs: "))
learning_rate = 0.01

# Create and train the neural network
model = NeuralNetwork(inputLayerSize, outputLayerSize)
for neurons in neurons_per_layer:
    model.add_hidden_layer(neurons)

model.train(X_train, y_train_encoded, epochs, learning_rate)

# Test the trained model
predictions_encoded = model.predict(X_test)

# Decode predictions back to original labels
predictions = np.argmax(predictions_encoded, axis=1)

# Decode one-hot encoded labels back to original labels for evaluation
y_test_decoded = np.argmax(y_test_encoded, axis=1)

# Map predictions and true labels back to original values
unique_labels = np.unique(y_train)
predictions_mapped = unique_labels[predictions]
y_test_decoded_mapped = unique_labels[y_test_decoded]

# Evaluate the model
accuracy = accuracy_score(y_test_decoded_mapped, predictions_mapped)
print(f'\n\nAccuracy: {accuracy}')

# Additional evaluation metrics
print('\nClassification Report:')
print(classification_report(y_test_decoded_mapped, predictions_mapped, labels=unique_labels))

# Additional evaluation metrics
print('\nConfusion Matrix:')
cm = confusion_matrix(y_test_decoded_mapped, predictions_mapped, labels=unique_labels)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()