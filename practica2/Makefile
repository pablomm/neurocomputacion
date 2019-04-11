# Jorge Arellano Subias
# Pablo Marcos Manchon

pp=python
perceptron=retro.py
data=data

all:
	# No se requiere compilacion
	# Para obtener ayuda ejecute retro_help o help

help:
	@echo Practica 2
	@echo Para ejecutar el perceptron utilize el script $(perceptron)
	@echo Debe ejecutarse empleando python3
	@echo Cuyo funcionamiento se muestra con el comando make retro_help
	@echo Este script entrena el perceptron y hace validacion de los datos
	@echo volcando la predicción en un fichero si así se especifica.

	@echo
	@echo Autores:
	@echo Jorge Arellano y Pablo Marcos

retro_help:
	$(pp) $(perceptron) -h

retro_exec2:
	$(pp) $(perceptron) $(data)/problema_real2.txt 70 -l 50 --epoch 300 --output prediccion_problema2.txt

retro_exec6:
	$(pp) $(perceptron) $(data)/problema_real6.txt 70 -l 20 --epoch 5000 --normalize normal --output prediccion_problema6.txt
