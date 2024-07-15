import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the dataset
X = np.array([[1, 2], [2, 3], [3, 3], [6, 6], [7, 7]])
y = np.array([0, 0, 1, 1, 1])

# Create and train the KNN model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# New data point
new_point = np.array([[4, 4]])
predicted_class = knn.predict(new_point)

# Plot the dataset and the new data point
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
plt.scatter(new_point[:, 0], new_point[:, 1], color='green', marker='x', s=100, label='New Point (Class {})'.format(predicted_class[0]))

# Plot the k nearest neighbors
distances, indices = knn.kneighbors(new_point)
for index in indices[0]:
    plt.plot([new_point[0][0], X[index][0]], [new_point[0][1], X[index][1]], 'k--')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title(f'K-Nearest Neighbors (k={k})')
plt.show()
