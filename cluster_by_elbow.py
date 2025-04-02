from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

processed_data_file = 'saved_file/processed_data_cluster.pkl'
k_cluster_label_file = 'saved_file/k_cluster_label.pkl'

with open(processed_data_file, 'rb') as f:
    processed_company = pickle.load(f)


#Calculate the 100 most relevant fourier transform coefficients
data = []
index = []

for key, item in processed_company.items():
    fourier_coef = fft(item['Close'].values)
    fourier_ls = [np.abs(coef) for coef in fourier_coef]
    fourier_ls.sort(reverse=True)
    data.append(fourier_ls[:100])
    index.append(key)
    
df_0 = pd.DataFrame(data, index=index)   

# Initialize lists to store distortion and inertia values
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
range_n_clusters = range(2,30)

# Fit K-means for different values of k
for k in range_n_clusters:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(df_0.values)
    
    # Calculate distortion as the average squared distance from points to their cluster centers
    distortions.append(sum(np.min(cdist(df_0.values, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / df_0.values.shape[0])
    
    # Inertia is calculated directly by KMeans
    inertias.append(kmeanModel.inertia_)
    
    # Store the mappings for easy access
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]

print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')
# Plotting the graph of k versus Distortion
plt.plot(range_n_clusters, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.grid()
plt.show()

print("Inertia values:")
for key, val in mapping2.items():
    print(f'{key} : {val}')
# Plotting the graph of k versus Inertia
plt.plot(range_n_clusters, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.grid()
plt.show()