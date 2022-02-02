import numpy as np
"""'
Previously we did single sample of input: (Three neurons with 4 inputs)
inputs = [1, 2, 3, 2.5] 
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]] 
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases #order of weights and inputs does matter.
#^ same as [np.dot(weights[0], inputs), np.dot(weights[1],inputs),  np.dot(weights[2], inputs)] <-- returns a list/vector

print(output)"""

#------------Batch Inputs----------#
"""
Why Batches?
You can calculate the data in parralel
batch size is 32 standardly
"""

#same amount of neurons just more batches
#3x4 matrix
inputs = [[ 1.0,  2.0,  3.0,  2.5],
          [ 2.0,  5.0, -1.0,  2.0], 
          [-1.5,  2.7,  3.3, -0.8]]

#no changes because no changes in number of neurons so no more synapses
#3x4 matrix
weights = [[ 0.2,   0.8,  -0.5,   1.0 ], 
           [ 0.5,  -0.91,  0.26,  -0.5],
           [-0.26, -0.27,  0.17,  0.87]] 

biases = [2, 3, 0.5]

#We can't multiply a 3x4 matrix with a 3x4 matrix. We have to transpose one matrix.

output = np.dot(inputs, np.array(weights).T) + biases  #np.dot normally makes 2D lists np.arrays but we need to do it manually to use the .T and transpose it.

#^ same as [np.dot(weights[0], inputs), np.dot(weights[1],inputs),  np.dot(weights[2], inputs)] <-- returns a list/vector


print(output)