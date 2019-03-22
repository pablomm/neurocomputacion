
import numpy as np
from . import RedNeuronal, sigmoidal_bipolar, derivada_sigmoidal_bipolar


class PerceptronMulticapa(RedNeuronal):
    def __init__(self, capas=[], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar,
                 title="Perceptron Multicapa"):

         #Suponer al menos una capa oculta??
        self.capas = capas
        self.derivada = derivada

        super().__init__(activacion=activacion, title=title)

    def _inicializar_matriz(self, n, m):
        return np.random.uniform(-.5, .5, size=(n, m))

    def _inicializar_capas(self):

        n_capas = len(self.capas) + 1
        self.n_capas = len(self.capas) + 1
        self.n_capas_ocultas = len(self.capas)

        #Matrices con pesos de capas ocultas
        self.pesos = np.empty(shape=n_capas, dtype=object)

        #Matrices de valores de deltas de cada peso
        self.deltas = np.empty(shape=n_capas, dtype=object)

        #Lista de valores de entradas a cada neurona
        self.yin = np.empty(shape=n_capas, dtype=object)
        self.errores = np.empty(shape=n_capas, dtype=object)

        #Lista de valores de salidas a cada neurona
        self.fyin = np.empty(shape=n_capas, dtype=object)


        for i in range(n_capas - 1):
            self.yin[i] = np.zeros(shape=self.capas[i])
            self.errores[i] = np.empty(shape=self.capas[i])
            self.fyin[i] = np.zeros(shape=self.capas[i]+1)
            self.fyin[i][-1] = 1

        self.yin[-1] = np.zeros(shape=self.n_salida)
        self.fyin[-1] = np.zeros(shape=self.n_salida + 1)
        self.fyin[i][-1] = 1


        # Caso sin capas ocultas
        if n_capas == 1:
            self.pesos[0] = self._inicializar_matriz(self.n_entrada+1, self.n_salida)
            self.deltas[0] = np.zeros((self.n_entrada+1, self.n_salida))
        else:
            # Capa de entrada
            self.pesos[0] = self._inicializar_matriz(self.n_entrada+1, self.capas[0])
            self.deltas[0] = np.zeros((self.n_entrada+1, self.capas[0]))

            # Capas intermedias
            for i in range(1, n_capas-1):
                self.pesos[i] = self._inicializar_matriz(self.capas[i-1]+1, self.capas[i])
                self.deltas[i] = np.zeros((self.capas[i-1]+1, self.capas[i]))

            # Capa de salida
            self.pesos[-1] = self._inicializar_matriz(self.capas[-1]+1, self.n_salida)
            self.deltas[-1] = np.zeros((self.capas[-1]+1, self.n_salida))


    def fit(self, X_train, y_train, learn_rate=.1, epoch=3):


        Xshape = X_train.shape
        yshape = y_train.shape

        if Xshape[0] != yshape[0]:
            raise ValueError("No coninciden el numero de datos")

        self.n_entrada = Xshape[1]
        self.n_salida = yshape[1]


        self._inicializar_capas()

        v_entrada = np.ones(self.n_entrada + 1)

        for _ in range(epoch):
            for x, y in zip(X_train, y_train):
                v_entrada[:-1] = x

                self.yin[0] = v_entrada @ self.pesos[0]
                self.fyin[0][:-1] = self.activacion(self.yin[0])

                # Propagamos hacia delante
                for i in range(1, self.n_capas):
                    self.yin[i] = self.fyin[i-1] @ self.pesos[i]
                    self.fyin[i][:-1] = self.activacion(self.yin[i])

                # Calculamos errores hacia atras (6.1)
                self.errores[-1] = (y - self.fyin[-1][:-1]) * self.derivada(self.fyin[-1][:-1])

                # Calculamos los deltas con producto matricial (6.2)
                self.deltas[-1] = learn_rate * (self.errores[-1][:, np.newaxis] @ (self.fyin[-2])).T

                for i in range(self.n_capas_ocultas, 1, -1):

                    # Calculamos los errores (7.1, 7.2)
                    self.errores[i-1] = self.errores[i] @ self.pesos[i].T
                    self.errores[i-1] = self.errores[i-1] * self.derivada(self.fyin[i-1][:-1])

                    # Calculamos los deltas (7.3)
                    self.deltas[i-1] = learn_rate * (self.errores[i-1][:, np.newaxis] @ self.fyin[i-2]).T


                # Calculamos los errores
                self.errores[0] = self.errores[1] @ self.pesos[1].T
                self.errores[0] = self.errores[1] * self.derivada(self.fyin[1][:-1])

                # Calculamos los deltas
                self.deltas[0] = learn_rate * (self.errores[0][:, np.newaxis] @ v_entrada).T

                # Actualizamos los pesos
                for i in range(self.n_capas):
                    self.pesos[i] += self.deltas[i]

    def z_in(self, X_test):

        salida = np.empty((X_test.shape[0], self.n_salida))
        v_entrada = np.ones(self.n_entrada + 1)

        for j,x in enumerate(X_train):
            v_entrada[:-1] = x

            self.yin[0] = v_entrada @ self.pesos[0]
            self.fyin[0][:-1] = self.activacion(self.yin[0])

            # Propagamos hacia delante
            for i in range(1, self.n_capas-1):
                self.yin[i] = self.fyin[i-1] @ self.pesos[i]
                self.fyin[i][:-1] = self.activacion(self.yin[i])

            self.yin[self.n_capas-1] = self.fyin[self.n_capas-2] @ self.pesos[self.n_capas-1]

            salida[j] = self.yin[self.n_capas-1]


    def evaluar(self, X_test):
        return self.z_in(X_test) >= 0













#
