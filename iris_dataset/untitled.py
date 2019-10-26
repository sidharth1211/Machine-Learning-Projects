
"""
Created on Tue Oct 15 20:23:45 2019

@author: commu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('IRIS.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
y= labelencoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train) 
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#calculating accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
# Visualising the Training set results
plt.figure()
sns.pairplot(dataset.drop("Id", axis=1), hue = "Species", size=3, markers=["o", "s", "D"])
plt.show()


