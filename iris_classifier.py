
"""
Created on Tue Oct 15 20:23:45 2019

@author: commu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('IRIS.csv')
dataset.columns =['sepal_length','sepal_width','petal_length','petal_width','species']
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
y= labelencoder.fit_transform(y)

#visualizing the dataset
#plt.figure()
#sns.pairplot(data= dataset, hue = 'species', size=3, markers=["o", "s", "D"])
#plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train) 
y_pred = classifier.predict(X_test)
#K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#SVM Kernal classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#Decision Tree Classifire
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="gini",random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#calculating accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)*100
print('Accuracy of the model is equal ' + str(round(ac, 2)) + ' %.')





