import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Data: [Length (cm), Weight (g)]
X = [
    [10, 50],   # Small
    [15, 100],  # Small
    [25, 200],  # Medium
    [30, 250],  # Medium
    [50, 600],  # Large
    [55, 700]   # Large
]
y = ["Small", "Small", "Medium", "Medium", "Large", "Large"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Plot tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=["Length", "Weight"], class_names=model.classes_, filled=True)
plt.show()
