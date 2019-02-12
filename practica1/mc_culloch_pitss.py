
import numpy as np

class McCullochPitss:

    def __init__(self, W, n_entrada=1, n_salida=1, umbral=1, title="McCulloch Pitts"):

        self.W = W
        self.title = title
        self.n_entrada = n_entrada
        self.n_salida = n_salida
        self.l = self.W.shape[0]

        if self.W.shape[1] != self.l:
            raise ValueError("W debe ser una matriz cuadrada")
        elif self.l < (n_entrada + n_salida):
            raise ValueError("W debe tener al menos tamaño {}".format((n_entrada + n_salida)))

        self.umbral = umbral

    def activacion(self, z):

        return np.greater_equal(z, self.umbral, out=z)

    def draw(self):

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            print("Para dibujar la red debe instalar networkx y matplotlib")
            print(e)

        Wint = self.W.astype(int).T

        #G = nx.from_numpy_matrix(Wint)

        G = nx.DiGraph(Wint)

        #  Diccionario con los pesos de los enlaces
        labels = nx.get_edge_attributes(G,'weight')

        # Lista con posiciones de nodos
        pos = dict(enumerate(labels.keys()))

        colors = (self.n_entrada * ['red'] +
                  (self.l - self.n_entrada - self.n_salida) * ['blue'] +
                  self.n_salida * ['green'])

        if np.isscalar(self.umbral):
            sesgo = " ($\\theta={}$)".format(self.umbral)
        else:
            sesgo=""


        plt.title(self.title + sesgo)

        """edges_colors = []

        for v in labels:
            x,y = v

            v = self.W[y,x]
            if v > 0:
                v = "green"
            elif v < 0:
                v = "red"
            else:
                v = "white"

            edges_colors.append(v)"""


        nx.draw(G,pos, with_labels=True, node_color=colors, arrows=True)


        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)



        plt.show()


    def evaluar(self, x, tiempo=1):
        # O(W Y^(t-1)) = Y^(t)

        # Array con inputs
        x = np.atleast_2d(x)

        if x.shape[1] != self.n_entrada:
            raise ValueError("Cada fila debe contener {} valores de entrada".format(self.n_entrada))

        # Matriz con valores de salida
        y = np.zeros((tiempo, self.n_salida))

        # Vector con estados de la red
        z = np.zeros((self.l, 1))

        #print("Entrada a procesar", x)
        #print("Matriz de pesos", self.W)

        for t in range(tiempo):
            # Inicializamos capa de entrada

            #print("Antes", z)
            z[0:self.n_entrada, 0] = x[t]

            #print("Entrada", x[t])
            #print("Con input", z)



            z = np.matmul(self.W,z)
            #print("Antes de actiacion", z)

            z = self.activacion(z)

            #print("Despues", z)

            ##print(y[t].shape,z[-self.n_salida:,0].shape)
            y[t, :] = z[-self.n_salida:,0]

        return y

W = np.zeros((14,14))

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




M = McCullochPitss(W.T,3,2, umbral=2)

x = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

print("Resultado", M.evaluar(x, 6))


M.draw()
