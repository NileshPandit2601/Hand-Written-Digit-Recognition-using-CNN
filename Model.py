import numpy as np

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    """
    Implements the neural network cost function and gradient calculation.

    Parameters:
    nn_params (np.ndarray): Flattened parameters for the neural network.
    input_layer_size (int): Number of features in the input layer.
    hidden_layer_size (int): Number of neurons in the hidden layer.
    num_labels (int): Number of output classes.
    X (np.ndarray): Input data of shape (m, input_layer_size).
    y (np.ndarray): Labels of shape (m,).
    lamb (float): Regularization parameter.

    Returns:
    J (float): The cost of the neural network.
    grad (np.ndarray): The gradients of the cost with respect to the parameters.
    """
    # Weights are split back to Theta1, Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                        (num_labels, hidden_layer_size + 1))

    # Forward propagation
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)  # Adding bias unit to first layer
    a1 = X
    z2 = np.dot(X, Theta1.T)
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for second layer
    a2 = np.append(one_matrix, a2, axis=1)  # Adding bias unit to hidden layer
    z3 = np.dot(a2, Theta2.T)
    a3 = 1 / (1 + np.exp(-z3))  # Sigmoid activation for third layer

    # Convert y labels to a one-hot encoded matrix
    y_vect = np.zeros((m, num_labels))
    for i in range(m):
        y_vect[i, int(y[i])] = 1

    # Cost function with regularization
    epsilon = 1e-10  # Small value to prevent log(0)
    J = (1 / m) * (np.sum(-y_vect * np.log(a3 + epsilon) - (1 - y_vect) * np.log(1 - a3 + epsilon))) + \
        (lamb / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    # Backpropagation
    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, Theta2)[:, 1:] * (a2[:, 1:] * (1 - a2[:, 1:]))  # Apply sigmoid derivative here

    # Gradient calculation with regularization
    Theta1_grad = (1 / m) * np.dot(Delta2.T, a1) + (lamb / m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    Theta2_grad = (1 / m) * np.dot(Delta3.T, a2) + (lamb / m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    # Unroll gradients
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

    return J, grad