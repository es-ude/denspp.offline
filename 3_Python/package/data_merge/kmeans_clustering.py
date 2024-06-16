import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score

# Angenommen, 'data' ist Ihr 48000x48 Array
data_load = np.load(r"C:\Users\Haris\git\CPS_Projekt\denspp.offline\3_Python\package\data_merge\_waveforms_as_one_array.npy")
data = data_load[0:500]

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 2. Optimale Anzahl von Clustern bestimmen (hier Elbow Method)
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Clusterzahl')
    plt.ylabel('SSE')
    plt.show()

# Find the optimal number of clusters
find_optimal_clusters(data_normalized, 20)

# 3. KMeans-Clustering anwenden (nach Bestimmung der optimalen Clusterzahl)
n_clusters = 5  # Setzen Sie die optimale Anzahl von Clustern hier
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_normalized)
labels = kmeans.labels_

# 4. t-SNE für die Reduktion der Dimensionalität auf 3D anwenden
tsne = TSNE(n_components=3, perplexity=30, learning_rate=1, random_state=42)
data_tsne = tsne.fit_transform(data_normalized)

# 5. 3D-Plot der Cluster erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_tsne[:, 0], data_tsne[:, 1], data_tsne[:, 2], c=labels, cmap='viridis')

# Legende hinzufügen
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)

# Achsen beschriften
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
# Berechnen Sie die Inertia

inertia = kmeans.inertia_

# Berechnen Sie den Silhouettenkoeffizienten
silhouette = silhouette_score(data, labels)

print("Inertia: ", inertia)
print("Silhouettenkoeffizient: ", silhouette)
plt.show()