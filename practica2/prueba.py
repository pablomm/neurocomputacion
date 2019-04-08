
from neuro import *
import numpy as np

np.random.seed(1)
red = PerceptronMulticapa(capas=[2], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar)
data = np.loadtxt('data/and.txt', skiprows = 1)
x = np.array([[-1, 1]])
y = np.array([[1]])
red.fit(x, y,learn_rate=.25, epoch = 1)
print(red.evaluar(x))
print(red.pesos)
