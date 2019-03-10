




"""Implementacion del problema de deteccion de patrones de subida y bajada
empleando una red de McCulloch-Pitss.

Argumentos del fichero:

python fichero_entrada.in tiempo fichero_salida.in

"""
import sys
import numpy as np

from neuro import Adaline, heaviside, parse_argv_data


def usage():
    print("Adaline")
    print("python adaline.py data_file [test_file | % test] "
          "[outputfile | stdout] [train params]")
    print("Modo 1: train y test en distintos ficheros:")
    print("\tpython perceptron.py file_train file_test file_out [train]")
    print("Modo 2: Indicar porcentaje de test (e.g. 80%)")
    print("\tpython perceptron.py file_data 80  file_out [train]")
    print("Modo 3: Todos los datos usados en train y test")
    print("\tpython perceptron.py file_data 100  file_out [train]")
    print("[train] son los argumentos opcionales learn_rate, epoch, ecm y tol")
    sys.exit()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        usage()

    X_train, y_train, X_test, y_test = parse_argv_data(sys.argv)

    # Solo codificamos una neurona, la otra sera complementaria
    #y_train = y_train[:,0]
    #y_test = y_test[:,0]

    if len(sys.argv) > 3:
        file_out = sys.argv[3]
    else:
        file_out = "stdout"

    if len(sys.argv) > 4:
        learn_rate = float(sys.argv[4])
    else:
        learn_rate = .1

    if len(sys.argv) > 5:
        epoch = int(sys.argv[5])
    else:
        epoch = 100

    if len(sys.argv) > 6:
        ecm = float(sys.argv[6])
    else:
        ecm = 0.

    if len(sys.argv) > 7:
        tol = float(sys.argv[7])
    else:
        tol = 0.

    red = Adaline()

    print("Número de datos de entrenamiento: ", len(X_train))
    print("Tasa de aprendizaje: ", learn_rate)
    print("Número maximo de épocas", epoch)
    print("Criterio de parada: ecm <", ecm)
    print("Criterio de parada: tol <", tol)
    print("Entrenando...", end="\r")

    e = red.fit(X_train, y_train, learn_rate=learn_rate, epoch=epoch, ecm=ecm, tol=tol)

    print("Entrenado en ", len(e), "epocas")
    if len(e)>1:
        print("Error cuadrático medio con datos de entrenamiento:", e[-1])

    res = red.evaluar(X_test).astype(int)


    aciertos = np.equal(res, y_test)

    print("Precision en los datos de test por neurona : {}/{}".format(aciertos.sum(axis=0),len(res)))
    print("Precision total: {}/{}".format(np.bitwise_and.reduce(aciertos, axis=1).sum(), len(res)))


    if len(sys.argv) > 3 and sys.argv[3] != "stdout":
        np.savetxt(file_out, res, fmt="%d")
        print("Datos volcados al fichero", file_out)
    else:
        print("Resultado:")
        print('\n'.join(' '.join(str(cell) for cell in row) for row in res))
