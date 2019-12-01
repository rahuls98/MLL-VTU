#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

#load data and assign headers
iris = load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['Sepal-Length','Sepal_Width','Petal_Length','Petal_Width']

#fit kmeans model
kmeans = KMeans(n_clusters = 3)
kmodel = kmeans.fit(x)

#fit em model
gmm = GaussianMixture(n_components=3)
gmm.fit(x)
gmm_labels = gmm.predict(x)

#print confusion matrices for both classifications
print("Kmeans algorithm:\n ", confusion_matrix(iris.target, kmodel.labels_))
print("\nEM algorithm:\n ", confusion_matrix(iris.target, gmm_labels))

#print scatter plots for iris target clusters and kmeans-em classifications 
colormap = np.array(['red','blue','green'])
plt.subplot(2,2,1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[iris.target], s=40)
plt.subplot(2,2,2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[kmodel.labels_], s=40)
plt.subplot(2,2,3)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[gmm_labels], s=40)
plt.show()