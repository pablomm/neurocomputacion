
import numpy as np
import numpy.matlib

from . import RedNeuronal, sigmoidal_bipolar, derivada_sigmoidal_bipolar

def scalar_dot(a, b):
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()

    return np.matrix(a*b)


class PerceptronMulticapa(RedNeuronal):
    def __init__(self, capas=[], activacion=sigmoidal_bipolar,
                 derivada=derivada_sigmoidal_bipolar,
                 title="Perceptron Multicapa"):

        self.capas = capas
        self.derivada = derivada
        self.mean =None
        self.std = None
        self.min = None
        self.max = None
        self.normalizar = False

        super().__init__(activacion=activacion, title=title)

    def _inicializar_matriz(self, n, m):
        return np.matrix(np.random.uniform(-.5, .5, size=(n, m)))

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
            self.yin[i] = np.matlib.zeros(shape=self.capas[i])
            self.errores[i] = np.matlib.zeros(shape=self.capas[i])
            self.fyin[i] = np.matlib.zeros(shape=self.capas[i])



        self.yin[-1] = np.matlib.zeros(shape=self.n_salida)
        self.fyin[-1] = np.matlib.zeros(shape=self.n_salida)


        # Caso sin capas ocultas
        if n_capas == 1:
            self.pesos[0] = self._inicializar_matriz(self.n_entrada+1, self.n_salida)
            self.deltas[0] = np.matlib.zeros((self.n_entrada+1, self.n_salida))
        else:
            # Capa de entrada

            self.pesos[0] = self._inicializar_matriz(self.n_entrada+1, self.capas[0])
            self.deltas[0] = np.matlib.zeros((self.n_entrada+1, self.capas[0]))

            # Capas intermedias
            for i in range(1, n_capas-1):
                self.pesos[i] = self._inicializar_matriz(self.capas[i-1]+1, self.capas[i])
                self.deltas[i] = np.matlib.zeros((self.capas[i-1]+1, self.capas[i]))

            # Capa de salida
            self.pesos[-1] = self._inicializar_matriz(self.capas[-1]+1, self.n_salida)
            self.deltas[-1] = np.matlib.zeros((self.capas[-1]+1, self.n_salida))

    def _normalizar(self, X):


        if self.normalizar == "normal":
            if self.mean is None:
                self.mean = X.mean(axis = 0)

            if self.std is None:
                self.std = np.std(X, axis=0)

            return (X - self.mean) / self.std

        elif self.normalizar == "bipolar":
            if self.min is None:
                self.min = np.min(X, axis = 0)

            if self.max is None:
                self.max = np.max(X, axis=0)

            return 2*(X - self.min) / (self.max - self.min) - 1

        else:
            return X



    def fit(self, X_train, y_train, learn_rate=1, epoch=100, error=-1, normalizar=None):

        #X_train = np.copy(X_train)
        y_train = np.copy(y_train)

        #X_train[X_train==0] = -1
        y_train[y_train==0] = -1

        ecm = np.zeros(epoch)
        tol = np.zeros(epoch)

        if normalizar is not None:
            self.normalizar = normalizar
            X_train = self._normalizar(X_train)


        Xshape = X_train.shape
        yshape = y_train.shape

        if Xshape[0] != yshape[0]:
            raise ValueError("No coninciden el numero de datos")

        self.n_entrada = Xshape[1]
        self.n_salida = yshape[1]


        self._inicializar_capas()

        v_entrada = np.ones(self.n_entrada + 1)

        for epoca in range(1, epoch + 1):
            print("Epoca", epoca, end="\r")

            for x, y in zip(X_train, y_train):

                v_entrada[:-1] = x

                self.yin[0] = v_entrada @ self.pesos[0]
                self.fyin[0] = self.activacion(self.yin[0])

                # Propagamos hacia delante
                for i in range(1, self.n_capas):
                    self.yin[i] = np.column_stack((self.fyin[i-1], [1])) @ self.pesos[i]
                    self.fyin[i] = self.activacion(self.yin[i])


                diferencia = (y - self.fyin[-1])
                # Acumulamos el error cuadratico medio
                ecm[epoca-1] += np.square(diferencia).sum()

                # Calculamos errores hacia atras (6.1)
                self.errores[-1] = scalar_dot(diferencia,
                                              self.derivada(self.fyin[-1]))

                # Calculamos los deltas con producto matricial (6.2)
                self.deltas[-1] = learn_rate * (self.errores[-1].T @ np.column_stack((self.fyin[-2], [1]))).T

                for i in range(self.n_capas_ocultas, 1, -1):

                    # Calculamos los errores (7.1, 7.2)
                    self.errores[i-1] = self.errores[i] @ self.pesos[i][:-1].T
                    self.errores[i-1] = scalar_dot(self.errores[i-1],
                                                   self.derivada(self.fyin[i-1]))

                    # Calculamos los deltas (7.3)
                    self.deltas[i-1] = learn_rate * (self.errores[i-1].T @ np.column_stack((self.fyin[i-2], [1]))).T


                # Calculamos los errores
                self.errores[0] = self.errores[1] @ self.pesos[1][:-1].T
                self.errores[0] = scalar_dot(self.errores[0],
                                             self.derivada(self.fyin[0]))
                # Calculamos los deltas
                self.deltas[0] = learn_rate * (self.errores[0].T @ np.matrix(v_entrada)).T

                # Actualizamos los pesos
                for i in range(self.n_capas):
                    self.pesos[i] += self.deltas[i]

            ecm[epoca-1] /= self.n_salida * Xshape[0]
            #print(ecm[epoca -1], error)
            if ecm[epoca -1] < error:
                break

        return epoca, ecm[:epoca]


    def z_in(self, X_test):

        X_test = np.copy(X_test)
        X_test[X_test==0] = -1

        if self.normalizar is not None:
            X_test = self._normalizar(X_test)

        salida = np.empty((X_test.shape[0], self.n_salida))
        v_entrada = np.ones(self.n_entrada + 1)

        for j,x in enumerate(X_test):
            v_entrada[:-1] = x


            self.yin[0] = v_entrada @ self.pesos[0]
            self.fyin[0] = self.activacion(self.yin[0])

            # Propagamos hacia delante
            for i in range(1, self.n_capas-1):
                self.yin[i] = np.column_stack((self.fyin[i-1], [1])) @ self.pesos[i]
                self.fyin[i] = self.activacion(self.yin[i])

            self.yin[self.n_capas-1] = np.column_stack((self.fyin[self.n_capas-2], [1])) @ self.pesos[self.n_capas-1]

            salida[j] = self.yin[self.n_capas-1]


        return salida

    def evaluar(self, X_test):
        return (self.z_in(X_test) >= 0).astype(int)













#
