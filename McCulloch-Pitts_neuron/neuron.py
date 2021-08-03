import numpy as np

# Implementing the McCulloch-Pitts artificial neuron
np.random.seed(seed=0)

# Step 1: generate a vector of inputs and a vector of weights
I = np.random.choice([0,1], 3) # generate random vector I, sampling from {0,1}
W = np.random.choice([-1,1], 3) # generate random vector W, sampling from {-1,1}
print(f'Input vector: {I}, Weight vector: {W}')

# Step 2: compute the dot product between the vector of inputs and weights
dot = I @ W
print(f'Dot product: {dot}')

# Step 3: Define the threshold activation function
def linear_threshold_gate(dot, T):
    '''Returns the binary threshold output'''
    if dot >= T:
        return 1
    else:
        return 0

# Step 4: Compute the output based on the threshold value
T = 1
activation = linear_threshold_gate(dot, T)
print(f'Activation: {activation}')
