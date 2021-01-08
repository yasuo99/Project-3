# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:21:00 2020

@author: Thanh
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from itertools import product

class Dbscan: 
    df = pd.DataFrame()
    def __init__(self, X_numerics):
            self.df = X_numerics
    def DetermineEpsAndMinSamples(self):
        eps_values = np.arange(8,12.75,0.25) # eps values to be investigated
        min_samples = np.arange(3,10) # min_samples values to be investigated
        DBSCAN_params = list(product(eps_values, min_samples))

        no_of_clusters = []
        sil_score = []

        for p in DBSCAN_params:
            DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1]).fit(X_numerics)
            no_of_clusters.append(len(np.unique(DBS_clustering.labels_)))
            sil_score.append(silhouette_score(X_numerics, DBS_clustering.labels_))

        tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
        tmp['No_of_clusters'] = no_of_clusters

        pivot_1 = pd.pivot_table(tmp, values='No_of_clusters', index='Min_samples', columns='Eps')

        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(pivot_1, annot=True,annot_kws={"size": 16}, cmap="YlGnBu", ax=ax)
        ax.set_title('Number of clusters')
        plt.show()
        #Dựa vào heatplot trên, số lượng clusters nằm trong khoảng 17 - 4.
        tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
        tmp['Sil_score'] = sil_score

        pivot_1 = pd.pivot_table(tmp, values='Sil_score', index='Min_samples', columns='Eps')

        fig, ax = plt.subplots(figsize=(18,6))
        sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
        plt.show()
        #Silhouette score cao nhất là 0.26 với eps=12.5 và min_samples=4.
    def Clustering(self, eps, min_samples):
        DBS_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.df)
        DBSCAN_clustered = self.df.copy()
        DBSCAN_clustered.loc[:,'Cluster'] = DBS_clustering.labels_ # append labels to points

        DBSCAN_clust_sizes = DBSCAN_clustered.groupby('Cluster').size().to_frame()
        DBSCAN_clust_sizes.columns = ["DBSCAN_size"]
        print(DBSCAN_clust_sizes)
        return DBSCAN_clustered
        
mall_data = pd.read_csv('Mall_Customers.csv') 
print(mall_data.describe());
# Bỏ cột mã khách hàng khỏi dataset
X = mall_data.drop('CustomerID', axis = 1) 
# Bỏ cột giới tính ra khỏi dataset
X = X.drop('Gender', axis = 1) 
# Xử lý dữ liệu NaN
X.fillna(method ='ffill', inplace = True) 

print(X.head()) 
corr, _ = pearsonr(mall_data['Age'], mall_data['Spending Score (1-100)'])

jp = (sns.jointplot('Age', 'Spending Score (1-100)', data=mall_data,
                    kind='reg')).plot_joint(sns.kdeplot, zorder=0, n_levels=6)

plt.text(0,120, 'Pearson: {:.2f}'.format(corr))
plt.show()

#Tạo dataset chứa các mẫu data để phân cụm
X_numerics = mall_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
dbscan = Dbscan(X_numerics)
#Xác định eps và min_samples

dbscan.DetermineEpsAndMinSamples()
    
#Gom cụm với eps = 12.5 và min_samples = 4

DBSCAN_clustered = dbscan.Clustering(eps = 12.5, min_samples = 4)
print(DBSCAN_clustered)
#visualize cluster
outliers = DBSCAN_clustered[DBSCAN_clustered['Cluster']==-1]

fig2, (axes) = plt.subplots(1,2,figsize=(12,5))

#Visualize theo thu nhập và % sử dụng
sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster', ax=axes[0], palette='Set1', legend='full', s=45)
#Visualize theo tuổi và % sử dụng
sns.scatterplot('Age', 'Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster', palette='Set1', ax=axes[1], legend='full', s=45)

axes[0].scatter(outliers['Annual Income (k$)'], outliers['Spending Score (1-100)'], s=5, label='outliers', c="k")
axes[1].scatter(outliers['Age'], outliers['Spending Score (1-100)'], s=5, label='outliers', c="k")
axes[0].legend()
axes[1].legend()
plt.setp(axes[0].get_legend().get_texts(), fontsize='10')
plt.setp(axes[1].get_legend().get_texts(), fontsize='10')

plt.show()
