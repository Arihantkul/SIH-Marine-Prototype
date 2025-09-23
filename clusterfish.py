import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data: [Length (cm), Weight (g)]
X = [
    [10, 50],
    [12, 60],
    [15, 100],
    [25, 200],
    [30, 250],
    [50, 600],
    [55, 700],
    [60, 800]
]

# Make 3 clusters (Small, Medium, Large automatically)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Cluster predictions
labels = kmeans.labels_
print("Cluster labels:", labels)

# Plot clusters
X = np.array(X)
plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red", marker="X", s=200, label="Centers")
plt.xlabel("Length (cm)")
plt.ylabel("Weight (g)")
plt.title("Fish Clustering (KMeans)")
plt.legend()
plt.show()
