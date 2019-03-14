
import sys

import numpy as np

try:
    # Hecho asi por la libreria matplotlib de los laboratorios
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Error al importar matplotlib, las funciones de  plot no estaran "
          "disponibles.")
    print(e)



def parse_argv_data(argv):
    """Lee los ficheros de datos de entrada del programa."""

    if len(argv) < 2:
        sys.exit()

    file_train = argv[1]

    with open(file_train) as train_file:
        # Leemos cabecera
        n, m = [int(a) for a in train_file.readline().split()]

    data = np.loadtxt(file_train, skiprows=1)

    X = data[:,:n]
    y = data[:,-m:]
    n_ejemplos = len(X)

    try:
        if len(argv) < 3:
            percentage = 100
        else:
            percentage = int(argv[2])

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

    except ValueError:
        X_train = X
        y_train = y

        test_file = argv[2]

        data_test = np.loadtxt(test_file, skiprows=1)
        X_test = data_test[:,:n]
        y_test = data_test[:,-m:]

    return X_train, y_train, X_test, y_test

def plot_ecm(error):

    plt.style.use("seaborn")
    plt.plot(error)
    plt.xlabel("Época")
    plt.ylabel("Error cuadrático medio")

# Funcion reutilizada de FAA con modificaciones
def plotModel(X, clase, clf, title=None):

    if X.shape[1] > 2:
        raise ValueError("Plot solo disponible para datasets con 2 dimensiones")
    x = X[:, 0]
    y = X[:, 1]

    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.


    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))



    Z = clf.evaluar(np.c_[xx.ravel(), yy.ravel()])

    for k in range(clf.n_entrada):

        plt.figure()
        ax = plt.gca()

        z = Z[:,k]


        z = z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplot(1, 1, 1)
        ax.contourf(xx, yy, z, cmap=cm, alpha=.8)
        ax.contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

        if clase is not None:
            ax.scatter(x[clase[:,k]==0], y[clase[:,k]==0], c='#FF0000')
            ax.scatter(x[clase[:,k]==1], y[clase[:,k]==1], c='#0000FF')
        else:
            ax.plot(x,y,'g', linewidth=3)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if title is not None:
            ax.title.set_text("$Z_{}$: ".format(k+1) + title)
        else:
            ax.title.set_text("$Z_{}$".format(k+1))


def matriz_confusion(pred, real):

    clases = np.unique(real)

    r"""Genera matriz de confusion"""

    matriz = np.zeros((len(clases),len(clases)))

    for i in clases:
        clasei = real == i

        for j in clases:
            matriz[j,i] = np.sum(pred[clasei] == j)

    return matriz
