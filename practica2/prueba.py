
from neuro import *
import numpy as np

np.random.seed(1)
red = PerceptronMulticapa(capas=[2], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar)
data = np.loadtxt('data/nor.txt', skiprows = 1)
x = np.array(data[:,:-2])
y = np.array(data[:,-2:])
red.fit(x, y,learn_rate=.25, epoch = 100)
print('Evaluo datos de entrenamiento')
print(red.evaluar(x))
print(red.pesos)
