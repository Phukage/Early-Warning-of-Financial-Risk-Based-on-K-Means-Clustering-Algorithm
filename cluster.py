import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.fft import fft
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


processed_data_file = 'saved_file/processed_data_cluster.pkl'

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


# Calculate the silhouette score
silhouette_threshold = 0.5
range_n_clusters = [i for i in range(7,8)] 

stop_flag = True
for time in range(10):
    for n_cluster in range_n_clusters:
        clusterer = KMeans(n_clusters=n_cluster)
        cluster_labels = clusterer.fit_predict(df_0.values)
        silhouette_avg = float(silhouette_score(df_0, cluster_labels))  
        # print(f"For n_clusters = {n_cluster}, the average silhouette_score is : {silhouette_avg}")

        continue_flag = False
        if silhouette_avg > silhouette_threshold:
            for i in range(n_cluster):
                if df_0[cluster_labels == i].shape[0] > (df_0.shape[0] / 1.5):
                    continue_flag = True
                    break
            if continue_flag:
                #print(continue_flag)
                continue
            
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(df_0.values, cluster_labels)

            y_lower = 10
            n_below_avg = 0
            for i in range(n_cluster):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()
                if ith_cluster_silhouette_values[-1] < silhouette_avg:
                    n_below_avg += 1

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_cluster)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title(f"The silhouette plot for the {n_cluster} clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            with open(f'output_cluster_high_sil/cluster_{n_cluster}_{silhouette_avg}.pickle', 'wb') as f:
                pickle.dump(cluster_labels, f)
            with open(f'output_cluster_high_sil/cluster_{n_cluster}_{silhouette_avg}_data.png', 'wb') as f:
                plt.savefig(f)
            stop_flag = False
            # plt.show()
    


    



