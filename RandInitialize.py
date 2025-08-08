import numpy as np

def initialise(num_units, num_inputs, epsilon=0.15):
    # Randomly initializes values of thetas between [-epsilon, +epsilon]
    weights = np.random.rand(num_units, num_inputs + 1) * (2 * epsilon) - epsilon  
    return weights