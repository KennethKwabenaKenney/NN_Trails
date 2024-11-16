# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:14:58 2024

@author: kenneyke
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QFileDialog

# Set TF_ENABLE_ONEDNN_OPTS environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#%% Parameters
numHidden1 = 800
numHidden2 = 500
learning_rate = 0.01
numEpochs = 10
activation_hidden1 = "relu"
activation_hidden2 = "relu"
activation_output = "softmax"  # Softmax for multiclass classification

#%% Functions
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

def custom_stratified_split(X, Y, test_size=0.2, random_state=None):
    """
    test_size for validation proportion sampling
    """
  
    # Combine X and Y into a single DataFrame for easier manipulation
    data = pd.concat([X, Y], axis=1)

    # Calculate the frequency of each target value
    target_counts = Y.value_counts()

    # Initialize lists to store train and validation indices
    train_indices = []
    val_indices = []

    # Iterate over each unique target value
    for target_value, count in target_counts.items():
        # Get indices corresponding to the current target value
        indices = data.index[data[Y.name] == target_value].tolist()

        # Calculate the number of samples to include in the validation set for this target value
        val_size = int(count * test_size)

        # Randomly select indices for validation set
        np.random.seed(random_state)
        selected_indices = np.random.choice(indices, val_size, replace=False)

        # Check if any target is not represented in validation due to small size
        if len(selected_indices) == 0:
            # Duplicate the target in validation to ensure representation
            selected_indices = np.random.choice(indices, 1, replace=False)
        
        # Add the selected indices to validation indices without removing from training
        val_indices.extend(selected_indices)

        # Add all indices for this target to training indices
        train_indices.extend(indices)

    # Remove any duplicates in validation set from training set
    train_indices = list(set(train_indices) - set(val_indices))

    # Split the data into training and validation sets based on the selected indices
    X_train, X_val = X.loc[train_indices], X.loc[val_indices]
    Y_train, Y_val = Y.loc[train_indices], Y.loc[val_indices]

    return X_train, X_val, Y_train, Y_val


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

#%% Data Preprocessing
# Example usage
# X_train = np.random.rand(100, 100)  # Example random data for X_train
# X_test = np.random.rand(50, 100)    # Example random data for X_test
# Y_train = np.random.choice([140100, 110, 210, 170400], size=(100,))
# Y_test = np.random.choice([140100, 110, 210, 170400], size=(50,))

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
Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class', 'Sub_class']   #Columns to exclude from the X data
X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)    
y_train = Train_data['Ext_class']

# Testing
Test_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class', 'Sub_class']   #Columns to exclude from the X data
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



# Label encoding
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

#%% Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numHidden1, activation=activation_hidden1),
    tf.keras.layers.Dense(numHidden2, activation=activation_hidden2),
    tf.keras.layers.Dense(len(np.unique(Y_train)), activation=activation_output)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%% Train  and predict

# Train the model
history = model.fit(X_train, Y_train_encoded, epochs=numEpochs, batch_size=32, 
                    validation_data=(X_train, Y_train_encoded), verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test_encoded)
print(f'Test Accuracy: {test_accuracy}')

# # Plot training history
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Make predictions
# predictions = model.predict(X_test)
# predicted_labels = np.argmax(predictions, axis=1)

# # Inverse transform the predicted labels to get back the original labels
# predicted_labels_original = label_encoder.inverse_transform(predicted_labels)

# # Calculate accuracy
# accuracy = accuracy_score(Y_test, predicted_labels_original)
# print(f'Accuracy: {accuracy}')

# # Generate classification report with actual labels
# print('\nClassification Report:')
# classification_report_original = classification_report(Y_test, predicted_labels_original)
# print(classification_report_original)

# # Generate confusion matrix with actual labels
# def plot_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     plt.figure(figsize=(9, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
    
#     # Adjust tick positions to center them in the middle of the boxes
#     plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels)
#     plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels)
    
#     plt.show()

# plot_confusion_matrix(Y_test, predicted_labels_original, labels=np.unique(Y_train))