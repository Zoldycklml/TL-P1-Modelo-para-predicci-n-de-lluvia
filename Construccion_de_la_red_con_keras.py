#Importar librerias
import numpy as np # linear algebra
import pandas as pd # Procesar datos, el archivo que utilizamos es .CSV 
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Importar archivo de excel
df = pd.read_csv("/content/seattleWeather_1948-2017.csv")

#Mostrar los primeros datos
df.head(20)

#Conversion columna RAIN si es TRUE dentro de rain se almacena 1, de otra manera un 0, el dataframe en el campo rain es booleano, mientras que en el dataframe de la columna RAIN observamos TRUE o FALSE
df['rain']=[1 if i==True else 0 for i in df['RAIN']]

#sklearn es una libreria de machinelearning, el .model_selection es para seleccionar un modelo de entrenamiento, importamos train_test_split
from sklearn.model_selection import train_test_split

#Sacamos las comumnas de df
df.columns

#Observar la infromacion, tipos de datos, si estàn o no vacios
df.info()

#Filtro valores no nulos
df.dropna(inplace=True)

#para nuestro proceso de aprendizaje la variable X va a contener las primeras 3 columnas, mientras que  Y almacena rain numerico
X=df[['PRCP', 'TMAX', 'TMIN']]
y=df[['rain']]

#Asignar los resultados de la funcion train_test_split, esos resultados son los que se utilizaran para entrenar la IA
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2, random_state=41)

#Los valores en X son escalados, se normalizan los valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)


#Aporte Cesar
#el modelo que se utilizarà es Sequential, se asigna en ann, se utiliza activacion tipo relu y sigmoide
ann  = Sequential()
ann.add(Dense(units= 32, activation = 'relu', input_dim=3))
ann.add(Dense(units= 16, activation = 'relu'))
ann.add(Dense(units= 1, activation = 'sigmoid'))
ann.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

#Para entrenar la red neuronal, como parametros pasamos xtrain ytrain que previamente los habiamos asignado, en lotes de 10 por epoca, con un total de 20 epocas, el verbose es para mostrar los resultados, una barra de progreso
ann.fit(xtrain,ytrain, batch_size=10, epochs=20, verbose= 1)

#Muestra la predicciòn 
Y_pred = ann.predict(xtest)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred]
print(Y_pred)
