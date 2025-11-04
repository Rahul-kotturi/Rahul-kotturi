import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display first few rows
df.head()
# Basic stats
print(df.describe())

# Pairplot to visualize potential clusters
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
k = 5  # Based on the elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8, 6))
colors = ['#FF5733', '#33FF57', '#3357FF', '#FFC300', '#8E44AD']

for i in range(k):
    plt.scatter(
        X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1],
        s=100, c=colors[i], label=f'Cluster {i+1}'
    )

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=300, c='yellow', label='Centroids', edgecolors='black'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()
df['Cluster'] = y_kmeans
df.head()
