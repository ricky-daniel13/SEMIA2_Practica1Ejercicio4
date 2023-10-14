import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class PMC:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inicializar con pesos random
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Calcular capa escondida
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Calcular capa de salida
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backprop(self, inputs, targets, output):
        # Calculo de error
        error = targets - output

        # Gradiente con derivada de sigmoide
        delta_output = error * self.sigmoid_derivative(output)

        # Error en capa escondida
        error_hidden = delta_output.dot(self.weights_hidden_output.T)

        # Gradiente de capa escondida
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_layer_output)

        # Actualizacion de peso y bias
        self.weights_hidden_output += self.hidden_layer_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += inputs.reshape(-1, 1).dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                output = self.forward(inputs[i])

                # Retropropagacion
                self.backprop(inputs[i], targets[i], output)

    def predict(self, inputs):
        predictions = []
        for i in range(len(inputs)):
            output = self.forward(inputs[i])
            predictions.append(output)
        return np.array(predictions)

def popNpArray(array, index):
    temp = np.copy(array[index])
    array[index] = array[array.shape[0]-1]
    array = array[:-1].copy()
    return temp, array

def setsEntrenamiento(setsAmm, porcPrueba, array):
    setCount = round(array.shape[0]/setsAmm)
    print("Set count: " + str(setCount))
    trainCount = round(setCount * porcPrueba)
    print("train count: " + str(trainCount))
    sets = []
    for i in range(setsAmm):
        setEntr=[]
        entr = np.zeros((trainCount, array.shape[1])).copy()
        prb = np.zeros((setCount-trainCount, array.shape[1])).copy()
        for j in range(setCount):
            temp, array = popNpArray(array, random.randint(0,array.shape[0]-1))
            if (j < trainCount):
                entr[j] = temp
            else:
                prb[j-trainCount]=temp
        setEntr.append(entr)
        setEntr.append(prb)
        sets.append(setEntr)
    return sets

                

# Especifica la ubicación del archivo CSV
print("#INICIO#")

archivo_csv = '.\Practica1\irisbin.csv'

datos = pd.read_csv(archivo_csv, header=None)
print("-----Preparando datos--------")
datosNp = np.array(datos.iloc[:, :])
for i in range(datosNp.shape[0]):
    if(datosNp[i][4]==-1):
        datosNp[i][4]=0
    if(datosNp[i][5]==-1):
        datosNp[i][5]=0
    if(datosNp[i][6]==-1):
        datosNp[i][6]=0

# print(datosNp.T[3])
#y = np.array(datosNp.T[3],ndmin=2).T
#print(y)
sets = setsEntrenamiento(1, (1/3)*2, datosNp.copy())

epochs = 1000

# Input data (X)
x = np.delete(sets[0][0], [4,5,6], 1)
# Etiquetas (Y)
y = np.delete(sets[0][0], [0,1,2,3], 1)

#Entrenando
print("-----Entrenando--------")
red = PMC(input_size=4, hidden_size=4, output_size=3, learning_rate=0.1)
red.train(x, y, epochs)

# Pruebas
print("-----Comprobando--------")

# Input data (X)
x_t = np.delete(sets[0][1], [4,5,6], 1)
# Etiquetas (Y)
y_t = np.delete(sets[0][1], [0,1,2,3], 1)

predictions = np.round(red.predict(x_t))

np.set_printoptions(suppress=True)

for i in range(x_t.shape[0]):
    print(str(x_t[i]) + ", pred: " + str(predictions[i]) + ", real: " + str(y_t[i]))


'''
plt.figure(figsize=(8, 6))
plt.scatter(x_t[:, 0], x_t[:, 1], c=predictions[:, 0], cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perceptron con Backpropagation')
plt.colorbar()
plt.show()'''


# print("sets: ")
# print(sets)

