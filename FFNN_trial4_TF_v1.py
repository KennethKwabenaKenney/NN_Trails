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
numHidden1 = 99
numHidden2 = 35
learning_rate = 0.022476269722488427
numEpochs = 180
activation_hidden1 = "sigmoid"
activation_hidden2 = "relu"
activation_output = "softmax"  # Softmax for multiclass classification

# Stratiffied KFold Cross Validation
n_splits = 2  # Number of folds
n_repeats = 2  # Number of repetitions
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
    print("Original distribution:", Counter(y_train))
    
    return X_train, y_train, X_test, y_test

#%% Data Preprocessing

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
Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   #Columns to exclude from the X data
X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)  
y_train = Train_data['Ext_class']
 
# Testing
Test_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   #Columns to exclude from the X data
X_test = Test_data.drop(columns=Test_cols_to_remove, axis=1)    
y_test = Test_data['Ext_class']

# reclassify labels
print("Select an option:\n1. Reclassify non-common labels\n2. Synchronize labels")
X_train, y_train, X_test, y_test = reclassify_labels(X_train, y_train, X_test, y_test)

# Label encoding
label_encoder = LabelEncoder()
combined_labels = pd.concat([y_train, y_test])
label_encoder.fit(combined_labels) 
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)


# Initialize the Repeated Stratified k-Fold Cross-Validation
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

#%% Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numHidden1, activation=activation_hidden1),
    tf.keras.layers.Dense(numHidden2, activation=activation_hidden2),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation=activation_output)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%% Train  and predict
for train_index, val_index in rskf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=numEpochs, batch_size=32, 
                        validation_data=(X_val_fold, y_val_fold ), verbose=1)

    # # Evaluate the model on the test data
    # test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    # print(f'Test Accuracy: {test_accuracy}')

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

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