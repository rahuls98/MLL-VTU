#import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load data and targets
iris = load_iris()
x = iris.data
y = iris.target

#split into train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#fit the knn model and predict targets
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Print predictions
print("Accuracy: ", accuracy_score(y_pred, y_test))
print("\n", confusion_matrix(y_pred, y_test))
