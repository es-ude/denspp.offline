import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Angenommen, 'data' ist Ihr 48000x48 Array
data = np.load(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data",
                                         "exp0_waveforms_as_one_array.npy"))

# Anzahl der Zeilen in Ihrem Array
num_rows = data.shape[0]

# Wählen Sie zufällig 100 Zeilenindizes aus
random_indices = np.random.choice(num_rows, size=2000, replace=False)

# Erstellen Sie ein neues Array mit den ausgewählten Zeilen
new_array = data[random_indices]

data = new_array

scaler = StandardScaler()
#data_normalized = scaler.fit_transform(data)

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

# 4. LDA für die Reduktion der Dimensionalität auf 3D anwenden
lda = LDA(n_components=3)
data_lda = lda.fit_transform(data_normalized, labels)

# 5. 3D-Plot der Cluster erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_lda[:, 0], data_lda[:, 1],  data_lda[:, 2], c=labels, cmap='viridis')

# Legende hinzufügen
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)

# Achsen beschriften
ax.set_xlabel('LDA 1')
ax.set_ylabel('LDA 2')
ax.set_zlabel('LDA 3')

# Berechnen Sie die Inertia
inertia = kmeans.inertia_

# Berechnen Sie den Silhouettenkoeffizienten
silhouette = silhouette_score(data, labels)

print("Inertia: ", inertia)
print("Silhouettenkoeffizient: ", silhouette)
plt.show()