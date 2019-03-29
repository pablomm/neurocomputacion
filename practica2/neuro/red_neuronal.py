#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modulo con la clase abstracta de redes neuronales y funciones de
transferencia.

"""

from abc import ABC, abstractmethod

import numpy as np

class RedNeuronal(ABC):
    """Clase general de una red neuronal"""

    def __init__(self, n_entrada=0, n_salida=0, activacion=None, title=""):
        self.n_entrada = n_entrada
        self.n_salida = n_salida
        self.activacion = activacion
        self.title = title

    @abstractmethod
    def evaluar(self, X):
        """Metodo para ejecutar la red con la entrada X

        Args:
            X (array_like): Matriz donde cada entrada representa un
                dato del conjunto de datos y cada columna una observacion.

        Returns:
            (numpy.ndarray): Matriz con los valores de la capa de salida
            correspondientes en cada fila.

        """
        pass

    @abstractmethod
    def z_in(self, X):
        """Metodo para ejecutar la red con la entrada X.

        Args:
            X (array_like): Matriz donde cada entrada representa un
                dato del conjunto de datos y cada columna una observacion.

        Returns:
            (numpy.ndarray): Devuelve los valores de la capa de salida sin
            aplicar la funcion de activacion en la capa de salida.

        """
        pass

    def __call__(self, X):
        """Metodo para ejecutar la red con la entrada X

        Args:
            X (array_like): Matriz donde cada entrada representa un
                dato del conjunto de datos y cada columna una observacion.

        Returns:
            (numpy.ndarray): Matriz con los valores de la capa de salida
            correspondientes en cada fila.

        """
        return self.evaluar(X)

    def _draw(self, W, integer_weights=True, pos=None):
        """Dibuja la red neuronal con sus pesos.
        Para ello utiliza la libreria matplotlib y networkx. Si no estan
        instaladas se lanzara una excepcion.

        Args:
            integer_weights (bool, opcional): Indica si los pesos son enteros
                para realizar la


        """
        # Importado de esta forma porque en los laboratorios de la
        # eps no esta correctamente instalado matplotlib y no puede
        # instalarse sin sudo
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            print("Para dibujar la red debe instalar networkx y matplotlib")
            raise e

        if integer_weights:
            W = W.astype(int)
        else:
            W = W.round(2)

        length = len(W)

        G = nx.DiGraph(W)

        #  Diccionario con los pesos de los enlaces
        labels = nx.get_edge_attributes(G,'weight')

        if pos is None:
            # Lista con posiciones de nodos
            pos = dict(enumerate(labels.keys()))


        colors = (self.n_entrada * ['red'] +
                  (length - self.n_entrada - self.n_salida) * ['blue'] +
                  self.n_salida * ['green'])

        if self.title is not None:
            plt.title(self.title)

        nx.draw(G,pos, with_labels=True, node_color=colors, arrows=True)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)


    def draw():
        """Metodo para dibujar la red neuronal."""
        raise NotImplementedError

# Implementacion de distintas funciones de transferencia

def heaviside(z_in, umbral=0, out=None):
    """ Funcion de activacion heaviside.
        1 si z_in >= umbral
        0 en otro caso

    """
    return np.greater_equal(z_in, umbral, out=out)


def sigmoidal(z_in):
    """Funcion sigmoidal
    sigma(z) = 1/(1+ e^-z_in)

    """
    return 1./(1. + np.exp(-z_in))

def sigmoidal_bipolar(z_in):
    """Funcion sigmoidal bipolar
    sigma(z) = 1/(1+ e^-z_in)

    """
    return 2./(1. + np.exp(-z_in)) - 1

def derivada_sigmoidal_bipolar(fx):
    """Derivada uncion sigmoidal bipolar
    sigma(z) = 1/(1+ e^-z_in)

    """
    return .5 * (1+fx) * (1 - fx)

'''
    def sigmoidal_bipolar(z_in, out):
        """Funcion sigmoidal bipolar
        sigma(z) = 1/(1+ e^-z_in)

        """
        z_in = np.mul(-1, z_in, out=out)
        np.exp(out, out=out)
        out += 1
        np.divide(2., out, out=out)
        out -= 1
        return

    def derivada_sigmoidal_bipolar(fx, out):
        """Derivada uncion sigmoidal bipolar
        sigma(z) = 1/(1+ e^-z_in)

        """
        np.sum(1, fx, out=out)
        out *= (1-fx)
        out *= .5
        return

'''
