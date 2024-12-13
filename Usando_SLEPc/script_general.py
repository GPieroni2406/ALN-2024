import subprocess
import os

# Configuración inicial
metodos = ["arnoldi", "krylov-schur"] # Métodos a ejecutar
tamaños_matriz = [10, 50, 100, 500, 1000, 2000]  # Tamaño de la matriz dispersa
archivo_matriz = "matrix.dat"  # Archivo para guardar la matriz generada
num_procesos = [8, 16]  # Número de procesos para paralelismo
resultados = {}

# Crear directorio para resultados
os.makedirs("resultados", exist_ok=True)

# Ejecutar el script generador de matriz
def generar_matriz(tamaño, nombre_archivo):
    print(f"Generando matriz de tamaño {tamaño}...")
    resultado = subprocess.run(
        ["python3", "generar_matriz.py", str(tamaño), nombre_archivo],
        capture_output=True,
        text=True
    )
    if resultado.returncode != 0:
        print("Error al generar la matriz:")
        print(resultado.stderr)
        raise RuntimeError("Error al generar la matriz")
    print(resultado.stdout)

# Ejecutar un método con la matriz generada
def ejecutar_metodo(metodo, n_procesos, archivo_matriz):
    archivo_script = f"{metodo}.py"
    archivo_salida = f"resultados/{metodo}_np{n_procesos}_resultados.txt"

    if n_procesos == 1:
        cmd = ["python3", archivo_script, archivo_matriz, archivo_salida]
    else:
        cmd = ["mpirun", "--use-hwthread-cpus", "-np", str(n_procesos), "python3", archivo_script, archivo_matriz, archivo_salida]

    subprocess.run(cmd, capture_output=False, text=True)


for tamaño_matriz in tamaños_matriz:
    # Generar la matriz
    generar_matriz(tamaño_matriz, archivo_matriz)

    # Ejecutar métodos
    for metodo in metodos:
        resultados[metodo] = {}
        for n_procesos in num_procesos:
            print(f"Ejecutando {metodo} con {n_procesos} procesos...")
            ejecutar_metodo(metodo, n_procesos, archivo_matriz)
            print()
        print()
    print()
    print()