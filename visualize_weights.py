import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load weights from Theta1.txt
Theta1 = np.loadtxt('Theta1.txt')

# Visualize the weights using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(Theta1, cmap='viridis')
plt.title('Visualization of Weights in Theta1')
plt.xlabel('Input Features')
plt.ylabel('Hidden Layer Neurons')
plt.show()
