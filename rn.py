# -*- coding: utf-8 -*-

#Procesamiento de datos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

##RED NEURONAL

import keras
from keras.models import Sequential
from keras.layers import Dense

#Iniciar la red

classifier = Sequential()

#Agregar input layer y la primera capa oculta
#esto es la capa oculta
#output_dim significa el numero de neuronas que tendrá la capa, se eligen 6 por una regla facil de
#dividir la suma de las entradas con las salidas entre dos.
#init se refiere a la forma en la que inicializaremos los pesos
#activation se refiere a la funcion de activacion
#input_dim se refiere a el numero de neuronas del input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#unicamente se omite el parametro input_dim porque automaticamente sabe de la capa anterior
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer
#se cambio output_dim porque este caso solo queremos precedir si se va o se queda
#se cambio la funcion de activacion de rectificadora a sigmoid(si fueran más de una salida se una softmax)
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#si la salida es binaria se usa binary_entropy y si es categorica se usa categorical_entropy
#metrics se refiere al metodo para ver coo funciona
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#entrenar la red
#batch_size es el numero de iteraciones que se deben hacer para que se actualicen los pesos
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#tenemos que pasar de probabilidad a boolean para checar que tan bueno fue
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)