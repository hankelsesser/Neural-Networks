import numpy as np

#dot product with 1 neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(inputs, weights) + bias #order of weights and inputs doesn't matter because they are both vectors
#print(output)

#dot product with 3 neurons
inputs = [1, 2, 3, 2.5] 
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]] 
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases #order of weights and inputs does matter.
#^ same as [np.dot(weights[0], inputs), np.dot(weights[1],inputs),  np.dot(weights[2], inputs)] <-- returns a list/vector

print(output)





#Calculate outputs without Numpy

# inputs = [1, 2, 3, 2.5] 

# weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]


# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases): # zips together each list of weights with their neurons bias
#     neuron_output = 0                                        
#     for n_input, weight in zip(inputs, neuron_weights): # zips together each weight with input
#         neuron_output += n_input*weight #adds the input and weight to the output of the neuron for each input and weight
#     neuron_output += neuron_bias 
#     layer_outputs.append(neuron_output)


# print(layer_outputs)

'''
Shape:
- List: Vector
    - inputs and biases
- 2D List : Matrix (Lists need to be homologous (same length))
   - Weights 
- 3D List 

- Tensor is an object that can be an array

dot product: 

a*b = [1, 2, 3]*[2, 4, 6] = 1*2 + 2*4 + 3*6 
- dot product of two vector gives you one value








'''