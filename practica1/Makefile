# Jorge Arellano Subias
# Pablo Marcos Manchon

pp=python
perceptron=perceptron.py
ada=adaline.py
perceptron_figuras=perceptron_figuras.py
ada_figuras=adaline_figuras.py
data=data

all:
	# No se requiere compilacion
	# Para obtener ayuda ejecute help, per_help o ada_help

help:
	@echo Practica 1
	@echo Makefile con ejemplos de ejecucion de los scripts
	@echo Para ejecutar el perceptron utilize el script $(perceptron)
	@echo Cuyo funcionamiento se muestra con el comando make per_help
	@echo Este script entrena el perceptron y hace validacion de los datos
	@echo mostrando la prediccion por pantalla o volcandola a un fichero
	@echo Para generar figuras con los resultados puede utilizarse de
	@echo forma analoga el script $(perceptron_figuras).
	@echo
	@echo El adaline se ejecuta de forma similar con los scripts $(ada)
	@echo y $(ada_figuras). Para obtener ayuda ejecute el comando make ada_help.
	@echo
	@echo Este makefile contiene instrucciones con ejemplos de ejecuciones.
	@echo
	@echo Autores:
	@echo Jorge Arellano y Pablo Marcos

per_help:
	$(pp) $(perceptron)

ada_help:
	$(pp) $(ada)

per_exec:
	# Utiliza 80% en entrenamiento y 20% en validacion y los params por defecto
	# y salida por stdout
	$(pp) $(perceptron) $(data)/problema_real2.txt 80

ada_exec:
	# Utiliza 80% en entrenamiento y 20% en validacion y los params por defecto
	# y salida por stdout
	$(pp) $(ada) $(data)/problema_real2.txt 80

# A continuacion algunos ejemplos de llamada a los scripts

per_exec_modo1:
	# Llamada al perceptron con modo 1 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# 80% de datos de train y 20% de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(perceptron) $(data)/problema_real1.txt 80 salida.txt 0.1 100 0.005 0.001 0.5

ada_exec_modo1:
	# Llamada al adaline con modo 1 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# 80% de datos de train y 20% de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(ada) $(data)/problema_real1.txt 80 salida.txt 0.1 100 0.005 0.001 0.5

per_exec_modo2:
	# Llamada al perceptron con modo 2 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# todos los de datos de train y de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(perceptron) $(data)/problema_real1.txt 100 salida.txt 0.1 100 0.005 0.001 0.5

ada_exec_modo2:
	# Llamada al adaline con modo 2 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# todos los datos de train y de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(ada) $(data)/problema_real1.txt 100 salida.txt 0.1 100 0.005 0.001 0.5

per_exec_modo3:
	# Llamada al perceptron con modo 3 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# se especifica el fichero de train y el de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(perceptron) $(data)/problema_real1.txt $(data)/problema_real1.txt salida.txt 0.1 100 0.005 0.001 0.5

ada_exec_modo3:
	# Llamada al adaline con modo 3 y especificando todos los parametros de
	# train, utilizando el problema real 1:
	# se especifica el fichero de train y el de test
	# Salida volcada en el archivo salida.txt
	# tasa de aprendizaje de 0.1
	# UN maximo de 100 epocas
	# Si el ecm<0.005 se detendra el aprendizaje
	# Si en dos iteraciones consecutives el peso que mas cambia varia menos de 0.001
	# tambien se detendra la ejecucion
	# Umbral de la funcion de activacion utilizada en entrenamiento igual a 0.5
	$(pp) $(ada) $(data)/problema_real1.txt $(data)/problema_real1.txt salida.txt 0.1 100 0.005 0.001 0.5

# A continuacion ejecucion de los problemas logicos viendo sus graficas
per_graph_and:
	$(pp) $(perceptron_figuras) $(data)/and.txt

per_graph_nand:
	$(pp) $(perceptron_figuras) $(data)/nand.txt

per_graph_nor:
	$(pp) $(perceptron_figuras) $(data)/nor.txt

per_graph_xor:
	$(pp) $(perceptron_figuras) $(data)/xor.txt

ada_graph_and:
	$(pp) $(ada_figuras) $(data)/and.txt

ada_graph_nand:
	$(pp) $(ada_figuras) $(data)/nand.txt

ada_graph_nor:
	$(pp) $(ada_figuras) $(data)/nor.txt

ada_graph_xor:
	$(pp) $(ada_figuras) $(data)/xor.txt
