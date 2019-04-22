# Jorge Arellano Subias
# Pablo Marcos Manchon


autoencoder=./Autoencoder


data=data
alfabeto=$(data)/alfabeto_dat.txt

all:
	# No se requiere compilacion
	# Para obtener ayuda ejecute retro_help o help

help:
	@echo Practica 3

	@echo
	@echo Autores:
	@echo Jorge Arellano y Pablo Marcos

autoencoder_help:
	$(autoencoder) -h

autoencoder_exec:
	@echo Creando datos de ejemplo
	./Crear_Alfabeto 20 0.13 $(alfabeto) alfabeto_train.txt --n_letras 15
	./Crear_Alfabeto 10 0.07 $(alfabeto) alfabeto_test.txt --n_letras 15
	$(autoencoder) alfabeto_train.txt alfabeto_test.txt -e 500 -l 100 --plot
