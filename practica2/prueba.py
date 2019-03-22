
from neuro import *

red = PerceptronMulticapa(capas=[3], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar)
data = np.loadtxt('data/and.txt', skiprows = 1)
x = data[:,:2]
y = data[:, 2:]
red.fit(x, y)
print(red.evaluar(x))
