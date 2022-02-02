import numpy as np

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# training input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1],
                            [0,1,0]])

# training output dataset
training_outputs = np.array([[0,1,1,0,0]]).T

#testing input dataset
testing_inputs =  np.array([[1,1,1],
                            [0,1,1],
                            [0,0,1],
                            [1,1,1]])

# testing output dataset
testing_outputs = np.array([[1,0,0,1]]).T

# seed random numbers to make calculation
np.random.seed(1)

#random starting weights
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(10000):

    # Define input layer for training
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    training_results = sigmoid(np.dot(input_layer, synaptic_weights))

    #get adjustments
    error = training_outputs - training_results
    adjustments = error * sigmoid_derivative(training_results)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(training_results)

# Define input layer for testing
input_layer = testing_inputs
# Normalize the product of the input layer with the synaptic weights
testing_results = sigmoid(np.dot(input_layer, synaptic_weights))

error = testing_outputs - testing_results
print("Error: ",error)
print(testing_outputs, "\n", testing_results)