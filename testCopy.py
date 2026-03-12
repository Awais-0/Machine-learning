import pandas as pd
import numpy as np

df = pd.read_csv(r'Datasets\placement.csv')

df = df.drop('Unnamed: 0', axis=True)

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)