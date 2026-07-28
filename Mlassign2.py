import os 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv('data.csv')  

print(df.describe())
print(df.info())
print(df.isnull().sum())
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

x = df.drop('diagnosis', axis=1)
y = LabelEncoder().fit_transform(df['diagnosis'])

print(x.head())
print(y)

#WITHOUT PCA

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy Score is : ", accuracy_score(y_test, y_pred))

#With PCA 
pca = PCA(n_components = 5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

model = LogisticRegression(max_iter=5000)
model.fit(x_train_pca , y_train)

pca_predict = model.predict(x_test_pca)

print("accuracy score adter reducing the dimensions is : ", accuracy_score(y_test, pca_predict))

 
#USING LDA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lda = LDA(n_components = 1)

model_lda = LogisticRegression(max_iter=5000)

model_lda.fit(x_train_scaled, y_train)

y_pred_lda = model_lda.predict(x_test_scaled)

print("Accuracy Score using LDA is : ", accuracy_score(y_test, y_pred_lda))