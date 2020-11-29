# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:21:00 2020

@author: Thanh
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans


X = pd.read_csv('Mall_Customers.csv') 

# Bỏ cột mã khách hàng khỏi dataset
X = X.drop('CustomerID', axis = 1) 

# Xử lý dữ liệu NaN
X.fillna(method ='ffill', inplace = True) 

print(X.head()) 

# Scaling dữ liệu
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Bình thường hóa dữ liệu
X_normalized = normalize(X_scaled) 

# Chuyển từ numpy array thành dataframe
X_normalized = pd.DataFrame(X_normalized) 

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 


plt.scatter(X_principal['P1'], X_principal['P2'])
plt.show()

X_principal = X_principal.sort_values(by=['P1','P2']);

df2 = pd.DataFrame(columns = ['index','distance'])
for i in range(0,len(X_principal) - 1):
    dist = np.linalg.norm(X_principal.iloc[i] - X_principal.iloc[i+1])
    df2 = df2.append({'index': str(i), 'distance': dist}, ignore_index=True)
    
df2 = df2.sort_values(by=['distance'])
plt.scatter(df2['index'], df2['distance'])
plt.title('Biểu đồ thể hiện khoảng cách giữa các điểm')
plt.show();
print("Tính toán")
df3 = df2[df2['distance'] < 0.5]
df3 = df3.sort_values(by=['distance'])
plt.scatter(df3['index'], df3['distance'])
plt.title('Biểu đồ thể hiện khoảng cách giữa các điểm mà có khoảng cách < 0.5')
plt.show();

#Lua chon eps tot nhat
# range_eps = [0.1,0.2,0.3,0.4,0.5]
# for i in range_eps:
#     print("eps value is: " + str(i))
#     db = DBSCAN(eps=i, min_samples=5).fit(X_principal)
#     core_sample_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_sample_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#     silhouette_avg = silhouette_score(X_principal,labels)
#     print("For eps value = "+str(i), "Diem trung binh silhouette_score la: ",silhouette_avg)

# #Lua chon so diem toi thieu cua 1 cluster
# min_samples = [1,2,3,4,5,6,7,8,9,10]
# for i in min_samples:
#     db = DBSCAN(eps=0.1, min_samples=i).fit(X_principal)
#     core_sample_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_sample_mask[db.core_sample_indices_] = True
#     labels = set([label for label in db.labels_ if label >= 0])
#     print("voi so diem toi thieu la " + str(i), "So cluster la: " + str(len(set(labels))))

db_default = DBSCAN(eps = 0.1, min_samples = 7).fit(X_principal) 
labels = db_default.labels_ 

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Ước tính số cluster là: %d' % n_clusters_)
print('Ước tính số điểm nhiễu là: %d' % n_noise_)

colours = {} 
colours[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[3] = 'yellow'
colours[4] = 'purple'
colours[-1] = 'k'


cvec = [colours[label] for label in labels] 


r = plt.scatter(X_principal['P1'], X_principal['P2'], color ='r'); 
g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g'); 
b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b'); 
k = plt.scatter(X_principal['P1'], X_principal['P2'], color ='k'); 
plt.show()


plt.figure(figsize =(9, 9)) 
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec) 

plt.legend((r, g, b, y,k), ('Nữ dưới 30 dùng trên 100', 'Nam dưới 30 dùng trên 100', 'Nam trên 30 dùng trên 200','Nữ trên 30 dùng trên 200', 'Còn lại'),scatterpoints = 1, 
		loc = 3, 
		ncol = 3, 
		fontsize = 8) 
plt.title('Biểu đồ thể hiện mức chi tiêu và thu nhập của nam và nữ ở độ tuổi từ 20 - 60')
plt.show() 
