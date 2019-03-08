

"""Implementacion del problema de deteccion de patrones de subida y bajada
empleando una red de McCulloch-Pitss.

Argumentos del fichero:

python fichero_entrada.in tiempo fichero_salida.in

"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from neuro import McCullochPitss, heaviside


def usage():
    print("python",sys.argv[0],"entrada [salida]")
    print("\tentrada: Nombre del fichero de entrada.")
    print("\tsalida: Fichero de salida, por defecto se imprime en stdout.")
    sys.exit()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        usage()

    entrada = sys.argv[1]
    x = np.loadtxt(entrada)

    tiempo = len(x) + 1

    # Matriz de pesos de la red de McCulloch Pitss
    W = np.zeros((14,14))

    # La posicion (i,j) representa el peso de la conexion i->j
    W[0,3] = 2
    W[1,4] = 2
    W[2,5] = 2
    W[3,6] = 1
    W[1,6] = 1
    W[4,7] = 1
    W[2,7] = 1
    W[5,8] = 1
    W[0,8] = 1
    W[6,12] = 2
    W[7,12] = 2
    W[8,12] = 2
    W[3,9] = 1
    W[2,9] = 1
    W[4,10] = 1
    W[0,10] = 1
    W[5,11] = 1
    W[1,11] = 1
    W[9,13] = 2
    W[10,13] = 2
    W[11,13] = 2

    # Umbral de la funcion de activacion
    umbral = 2

    # Construccion de la red, con un umbral = 2
    red = McCullochPitss(W, n_entrada=3, n_salida=2,
                    activacion=lambda z: heaviside(z, umbral))

    # Dibuja la red
    red.draw()
    plt.show()

    # Simula la red durante el tiempo indicado
    y = red(x, tiempo=tiempo)
    y = y.astype(int)

    if len(sys.argv) < 3:
        # Imprimimos igual que en el fichero de ejemplo
        print('\n'.join(' '.join(str(cell) for cell in row) for row in y))
    else:
        np.savetxt(y, sys.argv[3], fmt="%d")
