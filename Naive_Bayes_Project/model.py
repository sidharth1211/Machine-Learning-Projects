# Naive Bayes

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset[:,4]= dataset[:,4].map({'female': 1, 'male': 0})
val = pd.DataFrame(y,columns=['Purchased'])
val = val.replace({1:"Yes",0:"No"})
X = dataset.iloc[:, [2, 3]].values
y = val.iloc[:,:].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn import preprocessing
labelEncoder = le = preprocessing.LabelEncoder()
X[:,]

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

pickle.dump(classifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
