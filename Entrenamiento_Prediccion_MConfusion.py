#Para entrenar la red neuronal, como parametros pasamos xtrain ytrain que previamente los habiamos asignado, en lotes de 10 por epoca, con un total de 20 epocas, el verbose es para mostrar los resultados, una barra de progreso
ann.fit(xtrain,ytrain, batch_size=10, epochs=20, verbose= 1)

#Muestra la predicciòn 
Y_pred = ann.predict(xtest)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred]
print(Y_pred)

#Matriz de confusion, 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, Y_pred)
print(cm)
