
from neuro import *
import numpy as np

np.random.seed(1)
red = PerceptronMulticapa(capas=[1], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar)
data = np.loadtxt('data/and.txt', skiprows = 1)
x = np.atleast_2d(data[0,:2])
y = np.atleast_2d(data[0, 2:])
red.fit(x, y, epoch = 100)
print(red.evaluar(x))
print(red.pesos)
