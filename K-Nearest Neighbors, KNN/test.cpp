#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Define the structure for data points, including coordinates and labels
struct DataPoint {
    int x1;
    int x2;
    int label;
};

// Calculate the Euclidean distance between two points
double euclideanDistance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int main() {
    // Define the dataset
    std::vector<DataPoint> dataset = {
        {1, 2, 0},
        {2, 3, 0},
        {3, 3, 1},
        {6, 6, 1},
        {7, 7, 1}
    };

    // Define a new data point
    int new_x1 = 4;
    int new_x2 = 4;

    // Set the value of k
    int k = 3;

    // Find the k nearest neighbors
    std::vector<std::pair<double, int>> distances; // store pairs of distance and index

    for (size_t i = 0; i < dataset.size(); ++i) {
        double dist = euclideanDistance(new_x1, new_x2, dataset[i].x1, dataset[i].x2);
        distances.push_back(std::make_pair(dist, i));
    }

    // Sort distances
    std::sort(distances.begin(), distances.end());

    // Count occurrences of each label among the nearest neighbors
    int count_label0 = 0;
    int count_label1 = 0;

    for (int i = 0; i < k; ++i) {
        int index = distances[i].second;
        if (dataset[index].label == 0) {
            ++count_label0;
        }
        else {
            ++count_label1;
        }
    }

    // Predict the class of the new data point based on majority vote of nearest neighbors
    int predicted_class = (count_label0 > count_label1) ? 0 : 1;

    // Output the prediction result
    std::cout << "Predicted class for new point (" << new_x1 << ", " << new_x2 << "): " << predicted_class << std::endl;

    return 0;
}
