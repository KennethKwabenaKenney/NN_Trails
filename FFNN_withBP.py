# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PyQt5.QtWidgets import QApplication, QFileDialog

class NeuralNetwork:
    def __init__(self, inputLayerSize, outputLayerSize, hiddenSize1 = 800, hiddenSize2 = 500, learningRate = .01):    
        # hyperparameters
        self.hiddenLayer1Size = hiddenSize1
        self.hiddenLayer2Size = hiddenSize2
        self.learningRate = learningRate       
        self.inputLayerSize = inputLayerSize   # Work on its Function to Set this to the number of features/columns
        self.outputLayerSize = outputLayerSize   # Work on its function Set this to the number of target classes
        # weights
        self.inputToHidden1Weights = self.initWeightMatrix(self.inputLayerSize,self.hiddenLayer1Size)
        self.hidden1ToHidden2Weights = self.initWeightMatrix(self.hiddenLayer1Size,self.hiddenLayer2Size)
        self.hidden2ToOutputWeights = self.initWeightMatrix(self.hiddenLayer2Size,self.outputLayerSize)
       
    def initWeightMatrix(self, firstLayerSize, secondLayerSize):
        """Returns a matrix containing randomly initialized weights"""
        return np.random.normal(0, secondLayerSize**(-.5), size=(secondLayerSize, firstLayerSize))

    
    def train(self, X, Y_actual):
        """ 
        feedforward and backpropagate to update weight matrix """
        ######################## FORWARD PASS ########################
        # Sigmoid - 1st hidden layer
        Zh1 = np.dot(self.inputToHidden1Weights, X)
        Ah1 = sigmoid(Zh1)
        # Sigmoid - 2nd hidden layer
        Zh2 = np.dot(self.hidden1ToHidden2Weights, Ah1)
        Ah2 = sigmoid(Zh2)
        # SoftMax - output
        Zo = np.dot(self.hidden2ToOutputWeights, Ah2)
        Y = np.array(softmax(Zo))
        ######################## BACKWARD PASS ########################
        ## Error Matrices ##
        outputError = -(Y_actual - Y)
        secondHiddenError = np.dot(self.hidden2ToOutputWeights.transpose(), outputError)
        firstHiddenError = np.dot(self.hidden1ToHidden2Weights.transpose(), secondHiddenError)
        outputDelta = -(Y_actual - Y)*sigmoid_prime(Zo)
        secondHiddenDelta = secondHiddenError * sigmoid_prime(Zh2)
        firstHiddenDelta = firstHiddenError * sigmoid_prime(Zh1)
        ## OUTPUT LAYER TO HIDDEN LAYER 2 ##
        outputDelta = -(Y_actual - Y) * sigmoid_prime(Zo)
        outputLayerDeltaErrorWithRespectToWeight = np.dot(outputDelta, Ah2.transpose())
        ## HIDDEN LAYER 2 TO HIDDEN LAYER 1 ##
        HiddenLayer2DeltaErrorWithRespectToWeight = np.dot(secondHiddenDelta,Ah1.transpose())
        ## HIDDEN LAYER 1 TO INPUT LAYER ## 
        HiddenLayer1DeltaErrorWithRespectToWeight = np.dot(firstHiddenDelta,X.transpose())
        ######################## GRADIENT DESCENT ########################
        self.hidden2ToOutputWeights -= self.learningRate * (outputLayerDeltaErrorWithRespectToWeight)
        self.hidden1ToHidden2Weights -= self.learningRate * (HiddenLayer2DeltaErrorWithRespectToWeight)
        self.inputToHidden1Weights -= self.learningRate * (HiddenLayer1DeltaErrorWithRespectToWeight)

        
    def predict(self, X):
        """ Returns a tuple of [0] the predicted, [1] the confidence
        and [2] the raw one hot encoded Y prediction """
        #X = np.array(X).reshape(-1,1)
        ######################## FORWARD PASS ########################
        # Sigmoid - 1st hidden layer
        Zh1 = np.dot(self.inputToHidden1Weights, X)
        Ah1 = sigmoid(Zh1)
        # Sigmoid - 2nd hidden layer
        Zh2 = np.dot(self.hidden1ToHidden2Weights, Ah1)
        Ah2 = sigmoid(Zh2)
        # SoftMax - output
        Zo = np.dot(self.hidden2ToOutputWeights, Ah2)
        Y = np.array(softmax(Zo))
        prediction = (reverseOneHot(Y),round(np.amax(Y),3), Y)
        return prediction
        
    def test(self, X, Y):
        """ Tests the model on the given input """
        correct = 0
        total = 0
        for index, row in X.iterrows():
            prediction = self.predict(row)
            if (prediction[0] == reverseOneHot(Y[index])):
                correct += 1
            else:
                print("Incorrect prediction "+str(total-correct)+": Actual is: "+str(reverseOneHot(Y[index]))+"; Network prediction: "+str(prediction[:2])+", raw one hot prediction: \n"+str(prediction[2]))
            total += 1
        print('Accuracy of the FFNN on '+str(len(X))+' test: '+str(round((100 * correct / total),2))+'%')

        
def makeTarget(numpyArrayHot):
    """ Updates One Hot Encoded vector to target of .99 instead of 1 and
    .01 instead of 0 to eliminate vanishing gradient due to use of sigmoid
    activation function """
    numpyArray = numpyArrayHot.copy()        
    numpyArray[np.where(numpyArrayHot==np.max(numpyArrayHot))] = .99
    numpyArray[np.where(numpyArrayHot!=np.max(numpyArrayHot))] = .01
    return numpyArray

def reverseOneHot(numpyArrayHot):
    """ Takes in a one hot array and returns the index of one (Convert back) """
    return np.where(numpyArrayHot == np.max(numpyArrayHot))[0][0]

def sigmoid(x):
    """Apply sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """Apply softmax activation function"""
    return sigmoid(x) * (1 / sum(sigmoid(x)))

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
    y_train[~y_train.isin(common_labels)] = reclassification_label
    y_test[~y_test.isin(common_labels)] = reclassification_label
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
Train_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class']   #Columns to exclude from the X data
X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)    
y_train = Train_data['Ext_class']

# Testing
Test_cols_to_remove = ['In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 'Root_class']   #Columns to exclude from the X data
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

#%% Train the NN model
# Determine the number of features and unique targets
inputLayerSize = X_train.shape[1]
outputLayerSize = y_train.nunique()

NN = NeuralNetwork(inputLayerSize, outputLayerSize)

# Training process
print("Training Begins:")
for epoch in range(4):  # Assuming you still want to use 4 epochs
    #NN.test(X_test, y_test)
    print("Epoch number: "+str(epoch)+", continuing to train...")
    
    for index in range(len(X_train)):  # Make sure it iterates over all training data
        NN.train(X_train.iloc[index], y_train.iloc[index])

print("Training Completed")

#%% Test and Save Model

# Final testing after training is complete
print("Final test:")
NN.test(X_test, y_test)

# Save the model
pickle_out = open("FFNNs1.pkl", "wb") 
pickle.dump(NN, pickle_out)
pickle_out.close()

print("Model testing complete. Model saved.")