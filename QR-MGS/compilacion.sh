#!/bin/bash

# Nombre del ejecutable
EXECUTABLE="algoritmo_qr"

# Archivo fuente
SOURCE="src/qr_algorithm.cpp"

# Archivo de entrada de prueba
INPUT_MATRIX="data/matriz_1000.mtx"

# Compilar el programa con g++
echo "Compilando $SOURCE..."
g++ -std=c++17 -fopenmp -O3 -o $EXECUTABLE $SOURCE

# Verificar si la compilación fue exitosa
if [ $? -ne 0 ]; then
    echo "Error durante la compilación."
    exit 1
fi

echo "Compilación completada."
