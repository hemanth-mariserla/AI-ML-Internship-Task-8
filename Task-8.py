import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nVisualizing the raw data...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title('Initial Scatter Plot of Annual Income vs. Spending Score')
plt.savefig('initial_scatter_plot.png')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
range_k = range(1, 11)
for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range_k, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(range_k)
plt.grid(True)
plt.savefig('elbow_method.png')
print("Elbow Method plot saved as elbow_method.png\n")

optimal_k = 5
print(f"Fitting K-Means model with optimal K = {optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
df['Cluster'] = kmeans_final.labels_

print("Visualizing clusters...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
plt.title(f'Customer Segments (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.savefig('customer_segments.png')
print("Customer segments plot saved as customer_segments.png\n")

print("Evaluating clustering using Silhouette Score...")
score = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score: {score:.4f}")

print("\nFirst 5 rows of the dataset with cluster labels:")
print(df.head())
