N = 8  # size of the chessboard

def solveNQueens(board, col):
    if col == N:
        return True  # Found a solution
    for i in range(N):
        if isSafe(board, i, col):
            board[i][col] = 1
            if solveNQueens(board, col + 1):
                return True
            board[i][col] = 0
    return False

def isSafe(board, row, col):
    for x in range(col):
        if board[row][x] == 1:
            return False
    for x, y in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[x][y] == 1:
            return False
    for x, y in zip(range(row, N, 1), range(col, -1, -1)):
        if board[x][y] == 1:
            return False
    return True

board = [[0 for _ in range(N)] for _ in range(N)]
solveNQueens(board, 0)


###dfs
from collections import defaultdict

# This class represents a directed graph using adjacency list representation
class Graph:
    # Constructor
    def __init__(self):
        # Default dictionary to store graph
        self.graph = defaultdict(list)

    # Function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited):
        # Mark the current node as visited
        visited.add(v)

        # Recur for all the vertices adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses recursive DFSUtil()
    def DFS(self, v):
        # Create a set to store visited vertices
        visited = set()
        # Call the recursive helper function to do DFS traversal
        self.DFSUtil(v, visited)

# Driver code
if __name__ == "__main__":
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(2, 1)
    g.addEdge(2, 0)
    g.addEdge(1, 4)
    g.addEdge(1, 5)
    g.addEdge(4, 6)

    # DFS function call (no print output)
    g.DFS(0)


#####A*
from queue import PriorityQueue

graph = {
    'a': {'b': 4, 'c': 3},
    'b': {'f': 5, 'e': 12},
    'c': {'e': 10, 'd': 7},
    'd': {'e': 2},
    'e': {'z': 5},
    'f': {'z': 16},
}

heuristic = {
    'a': 14,
    'b': 12,
    'c': 11,
    'd': 6,
    'e': 4,
    'f': 11,
    'z': 0,
}

def a_star_search(graph, start, goal, heuristic):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in graph[current]:
            new_cost = cost_so_far[current] + graph[current][neighbor]
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                frontier.put((priority, neighbor))
                came_from[neighbor] = current

    return None

# Example usage
start = 'a'
goal = 'z'
path = a_star_search(graph, start, goal, heuristic)


### k-means
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()


####Block world program
from collections import deque

class BlockWorld:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def print_state(self, state):
        for i, stack in enumerate(state):
            print(f"Stack {i+1}: {stack}")

    def is_goal_state(self, state):
        return state == self.goal_state

    def get_next_states(self, state):
        next_states = []
        for i, stack in enumerate(state):
            if stack:  # If the current stack has any blocks
                block = stack[-1]  # Take the top block
                for j, other_stack in enumerate(state):
                    if i != j:  # Don't place it back in the same stack
                        # Create a deep copy of the current state
                        new_state = [s[:] for s in state]
                        new_state[i].pop()
                        new_state[j].append(block)
                        next_states.append(new_state)
        return next_states

    def solve(self):
        # Use BFS to explore possible block moves
        queue = deque([(self.initial_state, [self.initial_state])])
        visited = set()
        visited.add(tuple(tuple(stack) for stack in self.initial_state))

        while queue:
            state, path = queue.popleft()
            if self.is_goal_state(state):
                return path

            for next_state in self.get_next_states(state):
                next_state_tuple = tuple(tuple(stack) for stack in next_state)
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    queue.append((next_state, path + [next_state]))
        return None

# Example usage
initial_state = [["A"], ["B", "C"], []]
goal_state = [["C", "B", "A"], [], []]

block_world = BlockWorld(initial_state, goal_state)

print("Initial State:")
block_world.print_state(initial_state)

print("\nGoal State:")
block_world.print_state(goal_state)

solution = block_world.solve()

if solution:
    print("\nSolution:")
    for i, state in enumerate(solution):
        print(f"Step {i+1}:")
        block_world.print_state(state)
        print()
else:
    print("No solution found")


##SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and prepare data (Iris dataset)
data = datasets.load_iris()
X = data.data[:, :2]  # Use only sepal length and width
y = np.where(data.target == 0, 0, 1)  # Binary classification: Setosa vs Non-Setosa

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model with a linear kernel
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on the test set and print accuracy
y_pred = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, marker='o', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', label='Test')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('SVM Decision Boundary (Setosa vs Non-Setosa)')
plt.legend()
plt.show()



###ANN
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and prepare data (Iris dataset)
data = datasets.load_iris()
X = data.data[:, :2]  # Use only sepal length and width
y = np.where(data.target == 0, 0, 1)  # Binary classification: Setosa vs Non-Setosa

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))  # First hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Predict and evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plotting decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = (model.predict(grid_scaled) > 0.5).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, marker='o', label='Train Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', label='Test Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('ANN Decision Boundary (Setosa vs Non-Setosa)')
plt.legend()
plt.show()




##### Decision tree
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree

# Load the Iris dataset
data = load_iris()
X = data.data  # Features: sepal length, sepal width, petal length, petal width
y = data.target  # Target labels: species

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title('Decision Tree for Iris Dataset')
plt.show()


