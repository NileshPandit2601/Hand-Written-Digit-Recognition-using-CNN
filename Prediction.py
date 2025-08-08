import numpy as np

def softmax(z):
    """Compute softmax values for each class in the input array."""
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)  # Adding bias unit to first layer

    # Layer 2 (Hidden Layer)
    z2 = np.dot(X, Theta1.T)
    a2 = 1 / (1 + np.exp(-z2))  # Activation for second layer
    a2 = np.append(one_matrix, a2, axis=1)  # Adding bias unit to hidden layer

    # Layer 3 (Output Layer)
    z3 = np.dot(a2, Theta2.T)
    a3 = softmax(z3)  # Softmax activation for output layer

    p = np.argmax(a3, axis=1)  # Predicting the class based on max value of hypothesis
    return p