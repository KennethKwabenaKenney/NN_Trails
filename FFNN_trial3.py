# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:18:49 2024

@author: kenneyke
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QFileDialog

class NeuralNetwork:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, activation='sigmoid'):
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
        self.activation = activation
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        if self.activation == 'sigmoid':
            self.a2 = self.sigmoid(self.z2)
        elif self.activation == 'softmax':
            self.a2 = self.softmax(self.z2)
        else:
            raise ValueError("Invalid activation function")
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.softmax(self.z3)  # Using softmax for multi-class classification
    
    def costFunction(self, X, y):
        self.forward(X)
        J = -np.mean(np.sum(y * np.log(self.yHat), axis=1))  # Cross-entropy loss
        return J
    
    def costFunctionPrime(self, X, y):
        self.forward(X)
        delta3 = self.yHat - y
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        
        return dJdW1, dJdW2
    
    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            dJdW1, dJdW2 = self.costFunctionPrime(X, y)
            self.W1 -= learning_rate * dJdW1
            self.W2 -= learning_rate * dJdW2
            if epoch % 100 == 0:
                loss = self.costFunction(X, y)
                print(f'Epoch {epoch}: Loss={loss:.4f}')
    
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.yHat, axis=1)

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
        X_train, Y_train, X_test, Y_test = reclassify_labels(X_train, y_train, X_test, y_test)
elif action == '2':
    X_train, Y_train, X_test, Y_test = synchronize_labels(X_train, y_train, X_test, y_test, labels_file_path)
else:
    print("Invalid option entered. No changes made.")

# # Example usage code
# X_train = np.random.rand(100, 100)  # Example random data for X_train
# X_test = np.random.rand(50, 100)    # Example random data for X_test
# Y_train = np.random.choice([99, 110, 250, 500], size=(100,))
# Y_test = np.random.choice([99, 110, 250, 500], size=(50,))


inputLayerSize = X_train.shape[1]
outputLayerSize = len(np.unique(Y_train))

hiddenLayerSize = 10  # Adjust the number of neurons in the hidden layer as needed

# Get unique class labels from Y_train
unique_labels = np.unique(Y_train)

# Convert target data to one-hot encoding for multi-class classification
Y_train_encoded = pd.get_dummies(Y_train.squeeze()).values

# Initialize and train the neural network
nn = NeuralNetwork(inputLayerSize, hiddenLayerSize, outputLayerSize)
nn.train(X_train, Y_train_encoded)

# Make predictions on the test data
predictions = nn.predict(X_test)

# Decode predictions and Y_test for evaluation
Y_test_decoded = np.argmax(pd.get_dummies(Y_test.squeeze()).values, axis=1)
Y_test_decoded_mapped = unique_labels[Y_test_decoded]
predictions_mapped = unique_labels[predictions]

accuracy = accuracy_score(Y_test_decoded_mapped, predictions_mapped)
print(f'Accuracy Score: {accuracy:.4f}')

print('Classification Report:')
print(classification_report(Y_test_decoded_mapped, predictions_mapped))

# Calculate confusion matrix with labels
conf_matrix = confusion_matrix(Y_test_decoded_mapped, predictions_mapped, labels=unique_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()