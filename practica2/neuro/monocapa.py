#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np

from numpy.core.umath_tests import inner1d

from . import heaviside, RedNeuronal


class RedMonocapa(RedNeuronal):

    def __init__(self, activacion=heaviside, title="Red Monocapa"):



        super().__init__(n_entrada=0, n_salida=0, activacion=heaviside,
                         title=title)

    @abstractmethod
    def fit(self, X, y, learn_rate=None, epoch=None, ecm=0, tol=0,
            init_random=True):
        pass

    def draw(self):

        if self.n_entrada == 0:
            raise ValueError("La capa debe ser entrenada previamente")

        W = np.zeros((self.n_entrada + 1 + self.n_salida,
                      self.n_entrada + 1 + self.n_salida))

        for k in range(self.n_salida):
            W[k, -self.n_entrada:] = self.w[k]

        W[self.n_entrada, -self.n_salida:] = self.b

        pos = {}
        cte = self.n_entrada * self.n_salida
        for i in range(self.n_entrada + 1 + self.n_salida):
            if(i < self.n_entrada):
                pos[i] = (0, i*self.n_salida+1)
            elif i == self.n_entrada:
                pos[i] = (1, i*self.n_salida)
            else:
                pos[i] = (4, (i-self.n_entrada)*self.n_salida-1)

        self._draw(W, integer_weights=False, pos=pos)

    def z_in(self, X):
        X = np.atleast_2d(X)
        dot = np.array([self.b + np.matmul(self.w, x) for x in X])

        return dot

    def evaluar(self, X):

        return self.activacion(self.z_in(X))


class Perceptron(RedMonocapa):
    """Implementacion de un perceptron simple"""

    def __init__(self, umbral=0.5):
        self.umbral = umbral

        super().__init__(title="Perceptron")



    def activacion_entrenamiento(self, x):

        x = x.copy()
        if np.isscalar(x):
            if x > self.umbral:
                return 1
            elif x < -self.umbral:
                return -1
            return 0
        else:
            # Clase 1
            idx1 = x>self.umbral
            x[idx1] = 1

            # Clase -1
            idx2 = x<-self.umbral
            x[idx2] = -1

            # Region de incertibumbre
            x[np.bitwise_and(~idx1, ~idx2)] = 0

            return x


    def fit(self, X, y, learn_rate=None, epoch=None, ecm=0, tol=0, init_random=True):


        X = np.atleast_2d(X)
        # Crea copia del array de clases para cambiar a bipolar
        y = np.array(y).reshape(len(X), -1)

        self.n_entrada = X.shape[1]
        self.n_salida = y.shape[1]

        # Codificamos en forma bipolar para el entrenamiento
        y[y == 0] = -1

        # Numero de ejemplos y dimension de los datos
        n, m = X.shape

        if n != len(y):
            raise ValueError("La dimension de los datos de entrenamiento y sus "
                             "clases no coincide {} != {}".format(n, len(y)))

        if epoch is None:
            epoch = n

        if learn_rate is None:
            learn_rate = 1./n

        if init_random:
            b = np.random.uniform(-0.5,0.5, self.n_salida)
            w = np.random.uniform(-0.5,0.5,(self.n_salida, self.n_entrada))
        else:
            b = np.zeros(self.n_salida)
            w = np.zeros((self.n_salida, self.n_entrada))

        # Contadores de los criterios de convergencia
        i = 0
        error = ecm + 1
        errores = np.empty(epoch)
        current_tol = tol + 1

        # Critero de parada
        while i < epoch and error > ecm and current_tol > tol:

            error = 0
            w_anterior = w.copy()
            b_anterior = b.copy()

            # Iteramos sobre el conjunto de entrenamiento
            for j in range(n):

                y_in = b + np.matmul(w, X[j])

                fy_in = self.activacion_entrenamiento(y_in)
                #print(fy_in)


                for k in range(self.n_salida):
                    if fy_in[k] != y[j, k]:

                        alphat = learn_rate * y[j, k]
                        w[k] += alphat * X[j]
                        b[k] += alphat

                        # Sumamos error cuadratico medio
                        error += (y_in[k] - y[j,k])**2

            # Error cuadratico medio para el criterio de parada
            error /= n*self.n_salida
            errores[i] = error
            i += 1

            current_tol = max(np.max(np.abs(b_anterior-b)),
                              np.max(np.abs(w_anterior-w)))


        self.b = b
        self.w = w


        return errores[1:i]


class Adaline(RedMonocapa):
    """Implementacion de Adaline"""

    def __init__(self, umbral=0.5):
        self.umbral = umbral

        super().__init__(title="Adaline")

    def fit(self, X, y, learn_rate=None, epoch=None, ecm=0, tol=0,
            init_random=True):


        X = np.atleast_2d(X)
        # Crea copia del array de clases para cambiar a bipolar
        y = np.array(y).reshape(len(X), -1)

        self.n_entrada = X.shape[1]
        self.n_salida = y.shape[1]

        # Codificamos en forma bipolar para el entrenamiento
        y[y == 0] = -1

        # Numero de ejemplos y dimension de los datos
        n, m = X.shape

        if n != len(y):
            raise ValueError("La dimension de los datos de entrenamiento y sus "
                             "clases no coincide {} != {}".format(n, len(y)))

        if epoch is None:
            epoch = n

        if learn_rate is None:
            learn_rate = 1./n

        if init_random:
            b = np.random.uniform(-0.5,0.5, self.n_salida)
            w = np.random.uniform(-0.5,0.5,(self.n_salida, self.n_entrada))
        else:
            b = np.zeros(self.n_salida)
            w = np.zeros((self.n_salida, self.n_entrada))

        # Contadores de los criterios de convergencia
        i = 0
        error = ecm + 1
        errores = np.empty(epoch)
        current_tol = tol + 1

        # Critero de parada
        while i < epoch and error > ecm and current_tol > tol:

            error = 0
            w_anterior = w.copy()
            b_anterior = b.copy()

            # Iteramos sobre el conjunto de entrenamiento
            for j in range(n):

                y_in = b + np.matmul(w, X[j])

                for k in range(self.n_salida):

                    alphat = learn_rate * (y[j, k] - y_in[k])
                    w[k] += alphat * X[j]
                    b[k] += alphat

                    # Sumamos error cuadratico medio
                    error += (y_in[k] - y[j,k])**2

            # Error cuadratico medio para el criterio de parada
            error /= n*self.n_salida
            errores[i] = error
            i += 1

            current_tol = max(np.max(np.abs(b_anterior-b)),
                              np.max(np.abs(w_anterior-w)))


        self.b = b
        self.w = w


        return errores[1:i]
