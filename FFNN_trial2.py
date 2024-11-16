import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import seaborn as sns
#%% Parameters
numHidden1 = 500
numHidden2 = 300
learning_rate= .01
numEpochs = 500
"""Activation Function: sigmoid, relu, tanh"""
ActFnc_Hidden1 = "sigmoid"
ActFnc_Hidden2 = "sigmoid"

#%% FFNN with BP Class
class NeuralNetwork:
    def __init__(self, inputLayerSize, outputLayerSize, hiddenSize1, hiddenSize2, learningRate, activation1, activation2):
        # hyperparameters
        self.hiddenLayer1Size = hiddenSize1
        self.hiddenLayer2Size = hiddenSize2
        self.learningRate = learningRate
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        # Activation functions
        self.activation1 = activation1
        self.activation2 = activation2
        # weights
        self.inputToHidden1Weights = self.initWeightMatrix(self.inputLayerSize,self.hiddenLayer1Size)
        self.hidden1ToHidden2Weights = self.initWeightMatrix(self.hiddenLayer1Size,self.hiddenLayer2Size)
        self.hidden2ToOutputWeights = self.initWeightMatrix(self.hiddenLayer2Size,self.outputLayerSize)
        
    def initWeightMatrix(self, firstLayerSize, secondLayerSize):
        """Initializes weight matrices with small random values."""
        return np.random.randn(secondLayerSize, firstLayerSize) * np.sqrt(2 / firstLayerSize)

    def train(self, X, Y_actual, batch_size):
        # Convert pandas dataframes to numpy arrays if not already
        X = np.array(X)
        Y_actual = np.array(Y_actual)

        # Shuffle the data using the same indices for both X and Y_actual
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y_actual = Y_actual[indices]

        # Training using mini-batch gradient descent
        for start_idx in range(0, len(X) - batch_size + 1, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            Y_batch = Y_actual[start_idx:end_idx]
            
            """######################## FORWARD PASS ########################"""
            # Activation functions for hidden layer 1
            if self.activation1 == 'sigmoid':
                Ah1 = sigmoid(np.dot(self.inputToHidden1Weights, X_batch))
            elif self.activation1 == 'relu':
                Ah1 = relu(np.dot(self.inputToHidden1Weights, X_batch))
            elif self.activation1 == 'tanh':
                Ah1 = tanh(np.dot(self.inputToHidden1Weights, X_batch))
            else:
                raise ValueError("Invalid activation function for the 1st hidden layer")
            
            # Activation functions for hidden layer 2
            if self.activation2 == 'sigmoid':
                Ah2 = sigmoid(np.dot(self.hidden1ToHidden2Weights, Ah1))
            elif self.activation2 == 'relu':
                Ah2 = relu(np.dot(self.hidden1ToHidden2Weights, Ah1))
            elif self.activation2 == 'tanh':
                Ah2 = tanh(np.dot(self.hidden1ToHidden2Weights, Ah1))
            else:
                raise ValueError("Invalid activation function for the 2nd hidden layer")
            
            # SoftMax - output
            Y = softmax(np.dot(self.hidden2ToOutputWeights, Ah2))
            
            """######################## BACKWARD PASS ########################"""
            ## Error Matrices ##
            outputError = -(Y_batch - Y)
            secondHiddenError = np.dot(self.hidden2ToOutputWeights.transpose(), outputError)
            firstHiddenError = np.dot(self.hidden1ToHidden2Weights.transpose(), secondHiddenError)
            
            outputDelta = -(Y_batch - Y) * sigmoid_prime(Y)
            secondHiddenDelta = secondHiddenError * eval(self.activation2+'_prime')(Ah2)
            firstHiddenDelta = firstHiddenError * eval(self.activation1+'_prime')(Ah1)
            
            ## OUTPUT LAYER TO HIDDEN LAYER 2 ##
            outputLayerDeltaErrorWithRespectToWeight = np.dot(outputDelta,Ah2.transpose())
            ## HIDDEN LAYER 2 TO HIDDEN LAYER 1 ##
            HiddenLayer2DeltaErrorWithRespectToWeight = np.dot(secondHiddenDelta,Ah1.transpose())
            ## HIDDEN LAYER 1 TO INPUT LAYER ## 
            HiddenLayer1DeltaErrorWithRespectToWeight = np.dot(firstHiddenDelta,X_batch.transpose())
            ######################## GRADIENT DESCENT ########################
            self.hidden2ToOutputWeights -= self.learningRate * (outputLayerDeltaErrorWithRespectToWeight)
            self.hidden1ToHidden2Weights -= self.learningRate * (HiddenLayer2DeltaErrorWithRespectToWeight)
            self.inputToHidden1Weights -= self.learningRate * (HiddenLayer1DeltaErrorWithRespectToWeight)
    
    def predict(self, X):
        """Perform forward pass to make predictions."""
        # Forward pass
        Ah1 = eval(self.activation1)(np.dot(self.inputToHidden1Weights, X.T))
        Ah2 = eval(self.activation2)(np.dot(self.hidden1ToHidden2Weights, Ah1))
        Y_pred = softmax(np.dot(self.hidden2ToOutputWeights, Ah2))
        return Y_pred.T  # Transpose to match the shape of predictions_encoded in the example usage
            
#%% Support Functions           
def sigmoid(x):
    """Apply sigmoid activation function"""
    return 1/(1+np.exp(-1*x))

def relu(x):
    """Apply ReLU activation function"""
    return np.maximum(0, x)

def tanh(x):
    """Apply hyperbolic tangent (Tanh) activation function"""
    return np.tanh(x)

def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))

def relu_prime(x):
    """Derivative of the ReLU function."""
    return np.where(x <= 0, 0, 1)

def tanh_prime(x):
    """Derivative of the Tanh function."""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Apply softmax activation function"""
    return sigmoid(x) * (1/sum(sigmoid(x)))

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

#%%  
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
    
# Example usage code
# X_train = np.random.rand(100, 100)  # Example random data for X_train
# X_test = np.random.rand(50, 100)    # Example random data for X_test
# Y_train = np.random.choice([99, 110, 250, 500], size=(100,))
# Y_test = np.random.choice([99, 110, 250, 500], size=(50,))


inputLayerSize = X_train.shape[1]
outputLayerSize = len(np.unique(Y_train))

Y_train_encoded = pd.get_dummies(Y_train).values
Y_test_encoded = pd.get_dummies(Y_test).values

# Create and train the neural network 
model = NeuralNetwork(inputLayerSize, outputLayerSize, numHidden1, numHidden2, learning_rate, ActFnc_Hidden1, ActFnc_Hidden2)

for epoch in range(numEpochs):
    model.train(X_train, Y_train, inputLayerSize)

# Test the trained model
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
Y_test_decoded = np.argmax(Y_test_encoded, axis=1)

# Map predictions and true labels back to original values
unique_labels_train = np.unique(Y_train)
unique_labels_test = np.unique(Y_test)

predictions_mapped = unique_labels_train[predictions]
Y_test_decoded_mapped = unique_labels_train[Y_test_decoded]

# Evaluate the model
accuracy = accuracy_score(Y_test_decoded_mapped, predictions_mapped)
print(f'Accuracy: {accuracy}')

# Additional evaluation metrics
print('\nClassification Report:')
print(classification_report(Y_test_decoded_mapped, predictions_mapped, labels=unique_labels_train))

# Additional evaluation metrics
print('\nConfusion Matrix:')
cm = confusion_matrix(Y_test_decoded_mapped, predictions_mapped, labels=unique_labels_train)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=unique_labels_train, yticklabels=unique_labels_train)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
