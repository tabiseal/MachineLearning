K-Nearest Neighbors (KNN) is an instance-based learning algorithm used for classification and regression. The main idea of the KNN algorithm is to predict the label of a new data point by finding the $k$ nearest points (neighbors) in the training set and using their labels to determine the label of the new data point.

KNN Classification Algorithm Steps
Select parameter $k$: Choose the number of neighbors $k$.
Calculate distance: Compute the distance between the new data point and all points in the training set.
Select the nearest $k$ points: Find the $k$ closest points.
Vote: In classification tasks, the label of the new data point is predicted based on the majority label among the $k$ neighbors.
Simple Example
Suppose we have a two-dimensional dataset for classification:

\begin{tabular}{|c|c|c|}
\hline
$x_1$ & $x_2$ & Label \\
\hline
1 & 2 & 0 \\
2 & 3 & 0 \\
3 & 3 & 1 \\
6 & 6 & 1 \\
7 & 7 & 1 \\
\hline
\end{tabular}



We will use the KNN algorithm to predict the label of a new data point $(4,4)$.

Visualization using Python and Matplotlib
Below is a simple code example that demonstrates how to use KNN for classification and visualize the data points and the classification result of the new data point using Matplotlib.



### Results Analysis

After running the code above, we will see the following information:

1. **Nearest neighbors of the new data point:**：
    - Neighbor 1: Point [3, 3], class 1, distance $( \sqrt{(4-3)^2 + (4-3)^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.41 )$
    - Neighbor 2: Point [6, 6], class 1, distance $( \sqrt{(4-6)^2 + (4-6)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83 )$
    - Neighbor 3: Point [2, 3], class 0, distance $( \sqrt{(4-2)^2 + (4-3)^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.24 )$
2. **Plotting the graph**：The graph of the original data points and the new data point, where the new data point (4, 4) is marked with a green cross and connected to its 3 nearest neighbors with dashed lines.
3. **Predicted class:**：According to these 3 neighbors, the predicted class for the new data point (4, 4) is class 1, because out of the 3 neighbors, two are class 1 and one is class 0.

### Summary

According to the K-Nearest Neighbors algorithm and the above results, the predicted class for the new data point (4, 4) is 1. This is determined based on the chosen $k = 3$ and the classes of the nearest neighbors.


Specifically, with $k = 3$, the nearest neighbors to the new data point (4, 4) are:
1. Neighbor 1: Point [3, 3], class 1
2. Neighbor 2: Point [6, 6], class 1
3. Neighbor 3: Point [2, 3], class 0

Since two of the three nearest neighbors are class 1 and one is class 0, the predicted class for the new data point (4, 4) is class 1. This is achieved through a simple majority voting mechanism, where the majority class among the neighbors determines the class of the new data point.

### Algorithm Steps

1. **Calculate distances:**：
Calculate the Euclidean distance between the new data point (4, 4) and each point in the training set.
    The Euclidean distance formula is:
   
    
    $$
    d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
    $$
    
    Calculate specific distances:
    
    - The Euclidean distance formula is:
    
    $$
    d = \sqrt{(4-1)^2 + (4-2)^2} = \sqrt{3^2 + 2^2} = \sqrt{9 + 4} = \sqrt{13} \approx 3.61
    $$
    
    - Distance to (2, 3):
    
    $$
    d = \sqrt{(4-2)^2 + (4-3)^2} = \sqrt{2^2 + 1^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2.24
    $$
    
    - Distance to (3, 3):
    
    $$
    d = \sqrt{(4-3)^2 + (4-3)^2} = \sqrt{1^2 + 1^2} = \sqrt{1 + 1} = \sqrt{2} \approx 1.41
    $$
    
    - Distance to (6, 6):
    
    $$
    d = \sqrt{(4-6)^2 + (4-6)^2} = \sqrt{(-2)^2 + (-2)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83
    $$
    
    - Distance to (7, 7):
    
    $$
    d = \sqrt{(4-7)^2 + (4-7)^2} = \sqrt{(-3)^2 + (-3)^2} = \sqrt{9 + 9} = \sqrt{18} \approx 4.24
    $$
    
2. **Select the nearest k points:**：
Sort by distance and select the nearest 3 points:
    - (3, 3) distance ≈ 1.41, class 1
    - (6, 6) distance ≈ 2.83, class 1
    - (2, 3) distance ≈ 2.24, class 0
3. **Vote:**：
Vote on the classes of these 3 nearest neighbors:
    - Class 1: 2 votes
    - Class 0: 1 vote
4. **Predict the class:**：
The predicted class for the new data point (4, 4) is 1 because class 1 has the majority of votes.










