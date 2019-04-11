



"""Implementacion del problema de deteccion de patrones de subida y bajada
empleando una red de McCulloch-Pitss.

Argumentos del fichero:

python fichero_entrada.in tiempo fichero_salida.in

"""
import sys
import numpy as np
from neuro import PerceptronMulticapa, plotModel, plot_ecm, matriz_confusion

import argparse

def parse_data(file_train, output):
    """Lee los ficheros de datos de entrada del programa."""

    with open(file_train) as train_file:
        # Leemos cabecera
        n, m = [int(a) for a in train_file.readline().split()]

    data = np.loadtxt(file_train, skiprows=1)

    X = data[:,:n]
    y = data[:,-m:]

    n_ejemplos = len(X)

    # Caso fichero de test indicado como porcentaje
    if output.isdigit():
        percentage = int(output)

        if percentage < 100:

            corte = (n_ejemplos * percentage) // 100
            idx = np.arange(n_ejemplos)
            np.random.shuffle(idx)
            idx_train = idx[:corte]
            idx_test = idx[corte:]

            X_train = X[idx_train]
            y_train = y[idx_train]
            X_test = X[idx_test]
            y_test = y[idx_test]
        else:

            X_train = X_test = X
            y_train = y_test = y

    else:
        X_train = X
        y_train = y

        data_test = np.loadtxt(output, skiprows=1)

        #data_test = np.loadtxt(output, skiprows=1)
        X_test = data_test[:,:n]
        y_test = data_test[:,-m:]


    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

    # Parseador de argumentos del script
    parser = argparse.ArgumentParser(description='Perceptron multicapa')

    parser.add_argument('train', type=str, nargs=1, help='Datos de entrenamiento')
    parser.add_argument('test', type=str, nargs='?', default="100",
                        help='Archivo de salida / porcentage')
    parser.add_argument('-o', '--output', type=str, nargs=1, default=[None],
                       help="Fichero de salida")
    parser.add_argument('-e', '--epoch', type=int, nargs=1, default=[100],
                       help="Maximas épocas de entrenamiento")
    parser.add_argument('--learning', type=float, nargs=1, default=[.1],
                       help="Tasa de aprendizaje")
    parser.add_argument('--ecm', type=float, nargs=1, default=[0],
                       help="Condición de parada con error cuadrático medio")
    parser.add_argument('-l', '--layers', type=int, nargs='+', default=[10],
                       help="Capas ocultas")
    parser.add_argument('--normalize', type=str, nargs=1, default=[None],
                        help="Normalizar datos (min-max / normal)")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Realizar graficas")

    args = parser.parse_args()

    # Parseamos los ficheros de datos
    X_train, y_train, X_test, y_test = parse_data(args.train[0], args.test)

    print("Datos de entrenamiento:")
    print("\t", "Fichero", args.train[0])
    print("\t", "Maximo numero de épocas", args.epoch[0])
    print("\t", "Constante de aprendizaje", args.learning[0])
    print("\t", "Criterio de parada ecm < ", args.ecm[0])
    print("\t", "Capas ocultas", args.layers)
    print("\t", "Numero de datos de entrenamiento", len(X_train))
    print("\t", "Numero de datos de test", len(X_test))
    print("\t", "Normalizar", args.normalize[0])

    print(40 * "*")


    red = PerceptronMulticapa(capas=args.layers)

    # Entrenamos el perceptron, recopilamos informacion solo si vamos a plotear
    epoch, ecm = red.fit(X_train, y_train, learn_rate=args.learning[0],
                  epoch=args.epoch[0], normalizar=args.normalize[0], error=args.ecm[0])


    print("Entrenado en {} epocas".format(epoch))
    print("Error cuadrático medio en entrenamieno: {}".format(ecm[-1]))

    res = red.evaluar(X_train)
    y_test = y_test.astype(int)
    aciertos = np.equal(res, y_train)
    ac = aciertos.sum(axis=0)
    tot = len(res)

    print("Precision en los datos de train por neurona : {}/{} ({}%)".format(ac, tot, (100*ac / tot).round(2)))

    ac = np.bitwise_and.reduce(aciertos, axis=1).sum()
    print("Precision total en datos de train: {}/{} ({}%)".format(ac, tot, round(100* ac/tot,2)))

    res = red.evaluar(X_test)
    y_test = y_test.astype(int)
    aciertos = np.equal(res, y_test)
    ac = aciertos.sum(axis=0)
    tot = len(res)
    print("Precision en los datos de test por neurona : {}/{} ({}%)".format(ac, tot, (100*ac / tot).round(2)))

    ac = np.bitwise_and.reduce(aciertos, axis=1).sum()
    print("Precision total en datos de test: {}/{} ({}%)".format(ac, tot, round(100* ac/tot,2)))
    print("Matriz de confusion de datos de test: ")
    try:
        m = np.matrix(matriz_confusion(res, y_test))
        print(m)
    except:
        print("No se ha podido calcular la matriz de confusión.")

    if args.output[0] is not None:
        print("Volcando datos en archivo", args.output[0], end="\r")
        with open(args.output[0], "w") as f:
            f.write(str(X_test.shape[1]) + " " + str(res.shape[1]) + "\n")
            np.savetxt(f, np.hstack((X_test, res)))

        print(50 * " ", end="\r")
        print("Datos volcados en", args.output[0])




    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plot_ecm(ecm)
        # Dibujamos region de decision para R2
        if X_train.shape[1] == 2:
            plotModel(X_train, y_train, red)

        plt.show()
