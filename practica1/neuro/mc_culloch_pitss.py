#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from . import heaviside, RedNeuronal


class McCullochPitss(RedNeuronal):
    """Red general con todas las neuronas interconectadas"""

    def __init__(self, W, n_entrada=1, n_salida=1, activacion=heaviside, title=None):
        """Crea una red neuronal totalmente interconectada a partir de la matriz
        de pesos de sus conexiones. La red neuronal no tiene bias de entrada
        en las neuronas.

        La matriz de pesos W contendra en la entrada (i,j) el peso de la
        conexion entre la neurona i y j.

        Es implementada como un grafo, utilizando productos con la matriz de
        adyacencia para avanzar los pesos.

        Args:
            W (numpy.ndarray): Matriz cuadrada con las conexiones de la red.
            n_entrada (int, opcional): Numero de neuronas de la capa de entrada.
                Por defecto a 1.
            n_salida (int, opcional): Numero de neuronas de la capa de salida.
                Por defecto a 1.
            activacion (funcion, opcional): Funcion de activacion que recibira
                un array con las neuronas a ser activadas. Por defecto es usada
                la funcion Heaviside.
            title (str, opcional): Nombre de la red para utilizar en la funcion
                draw().
        """

        self.W = W.astype(float).T
        self.l = self.W.shape[0]

        if self.W.shape[1] != self.l:
            raise ValueError("W debe ser una matriz cuadrada")

        elif self.l < (n_entrada + n_salida):
            raise ValueError("W debe tener al menos tamaño {}".format((n_entrada + n_salida)))

        if title is None:
            title = "Red de McCulloch-Pitss"

        super().__init__(n_entrada=n_entrada, n_salida=n_salida,
                         activacion=activacion, title=title)


    def draw(self, integer_weights=True):
        """Dibuja la red neuronal con sus pesos.
        Para ello utiliza la libreria matplotlib y networkx. Si no estan
        instaladas se lanzara una excepcion.

        Args:
            integer_weights (bool, opcional): Indica si los pesos son enteros
                para realizar la


        """
        self._draw(self.W.T, integer_weights=integer_weights)


    def evaluar(self, x, tiempo=1):
        # O(W Y^(t-1)) = Y^(t)

        # Array con inputs
        x = np.atleast_2d(x)

        if x.shape[1] != self.n_entrada:
            raise ValueError("Cada fila debe contener {} valores de "
                             "entrada".format(self.n_entrada))

        # Matriz con valores de salida
        y = np.zeros((tiempo, self.n_salida))

        # Vector con estados de la red
        z = np.zeros((self.l, 1))

        n_values = len(x)

        for t in range(tiempo):
            # Inicializamos capa de entrada

            if (t < n_values):
                z[0:self.n_entrada, 0] = x[t]
            else:
                z[0:self.n_entrada, 0] = 0

            z = np.matmul(self.W,z)

            z = self.activacion(z)

            y[t, :] = z[-self.n_salida:,0]

        return y

    def z_in(self, x, tiempo=1):
        return self.evaluar(x, tiempo=1)

    def __call__(self, x, tiempo=1):
        return self.evaluar(x, tiempo=tiempo)
