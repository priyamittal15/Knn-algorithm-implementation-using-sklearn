import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

dataset = pd.read_csv(r'C:\Users\Priya Mittal\Documents\iris dataset01.csv')
dataset

feature_columns = ['sepal length in cm', 'sepal width in cm', 'sepal length in cm','sepal width in cm']
X = dataset[feature_columns].values
y = dataset['class'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24) # 80% training and 20% test

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=15)  # when (k==15).
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy = ' + str(round(accuracy, 2)) + ' %.')
