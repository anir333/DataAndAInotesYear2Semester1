"""
K-Means Clustering:
    1. Choose number of clusters
    2. Specify cluster seeds
    3. Assign each point to a centroid


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('csv/3.01. Country clusters.csv')
print(data.head())

X = data[['Latitude', 'Longitude']]
print(X)

kmeans = KMeans(2) # number of clusters
kmeans.fit(X)

identify_clusters = kmeans.fit_predict(X)
print(identify_clusters)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identify_clusters
print(data_with_clusters)

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()







data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})
print(data_mapped)
print(data['Language'].unique())
print(data_mapped['Language'].unique())


x = data_mapped[['Language']]

kmeans = KMeans(3) # number of clusters
kmeans.fit(x)

identify_clusters = kmeans.fit_predict(x)
print(identify_clusters)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identify_clusters
print(data_with_clusters)

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()
