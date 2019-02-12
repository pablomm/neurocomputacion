
import numpy as np

def escalon(z):
    return z

def sigmoidal(z):
    return 1./(1+np.exp(-z))


class Capa:
    def __init__(self, W, sesgo=.5, activacion=escalon):

        # Guarda la matriz de pesos traspuesta por eficiencia
        self.W = np.atleast_2d(W).T
        self.sesgo=sesgo
        self.activacion = activacion

    @property
    def shape(self):
        """(n_entrada, n_salida)"""
        return self.W.shape

    def __repr__(self):
        return "Capa:\n" + str(W)

    def evaluar(self, x):
        """Calcula el valor de activacion de las neuronas

        Args:
            x (numpy.ndarray): Vector fila con los valores de la capa anterior

        """
        ("El producto", x, W)
        z = x @ self.W

        ("Antes de activacion", z)
        z = self.activacion(z)

        a = z >= self.sesgo
        a = a.astype(float)
        ("Valor evaluar", a)

        return a



class RedNeuronal:

    def __init__(self, capas):

        self.capas = capas
        self.n_capas = len(capas)

        self.n_entrada = capas[0].shape[0]

        # Comprobacion concuerdan dimensiones de las capas
        n_anterior = capas[0].shape[1]

        for capa in capas[1:]:

            shape = capa.shape
            if shape[0] != n_anterior:
                raise ValueError("Tamaño de las capas incorrecto")

            n_anterior = shape[1]

        self.n_salida = n_anterior

    def evaluar(self, x, tiempo=1):

        x = np.atleast_2d(x)

        # Comprobacion datos de entrada
        if x.shape[1] != self.n_entrada:
            raise ValueError("Tamaño de la entrada incorrecto")

        elif x.shape[0] < tiempo:
            x2 = np.zeros((tiempo, x.shape[1]))
            x2[:x.shape[0]] = x
            x = x2

        # Matriz con valores de la salida
        y = np.empty((tiempo, self.n_salida))


        # Lista de estados de las capas
        z = np.empty(self.n_capas + 1, dtype=object)

        # Estado de la capa de entrada
        z[0] = np.empty(self.capas[0].shape[0])

        # Estado del resto de capas internas
        for i, capa in enumerate(self.capas):
            z[i+1] = np.zeros(capa.shape[1])

        # Iteracion principal
        for t in range(tiempo):

            # Valor de la entrada en tiempo t
            z[0] = x[t]

            # Actualizamos los pesos propagando hacia atras
            for i, capa in enumerate(self.capas[::-1]):

                z[self.n_capas - i] = capa.evaluar(z[self.n_capas - i - 1])

            # Actualizamos valor de la salida en tiempo t
            y[t] = z[-1]


        return y

    def __call__(self, x, tiempo=1):
        return self.evaluar(x, tiempo=tiempo)




"""
capa_1 = Capa(np.array([[2,-1],[-1,2]]),2)
capa_2 = Capa(np.array([2,2]),2)


red = RedNeuronal([capa_1, capa_2])



valores = [[1,1], [0,1],[1,0], [0,0]]

#print(red(valores, 5))
"""
