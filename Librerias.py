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
