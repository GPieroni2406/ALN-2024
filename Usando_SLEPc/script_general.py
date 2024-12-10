import subprocess
import os
import time

# Configuración inicial
methods = ["arnoldi", "kyrlov-schur", "lanczos"]
matrix_size = 10000  # Tamaño de la matriz dispersa
matrix_file = "matrix.dat"  # Archivo para guardar la matriz generada
process_counts = [1, 2, 4, 8]  # Número de procesos para paralelismo
results = {}

# Crear directorio para resultados
os.makedirs("results", exist_ok=True)

# Ejecutar el script generador de matriz
def generate_matrix(size, filename):
    print(f"Generando matriz de tamaño {size}...")
    result = subprocess.run(
        ["python3", "generar_matriz.py", str(size), filename],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error al generar la matriz:")
        print(result.stderr)
        raise RuntimeError("Error al generar la matriz")
    print(result.stdout)

# Ejecutar un método con la matriz generada
def execute_method(method, n_processes, matrix_file):
    script_file = f"{method}.py"
    output_file = f"results/{method}_np{n_processes}_results.txt"

    if n_processes == 1:
        cmd = ["python3", script_file, matrix_file, output_file]
    else:
        cmd = ["mpirun", "-np", str(n_processes), "python3", script_file, matrix_file, output_file]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    time_file = f"results/{method}_np{n_processes}_time.txt"
    with open(time_file, "w") as f:
        f.write(f"Tiempo de ejecución: {elapsed_time:.4f} segundos\n\n")

    return elapsed_time


# Generar la matriz
generate_matrix(matrix_size, matrix_file)

# Ejecutar métodos
for method in methods:
    results[method] = {}
    for n_processes in process_counts:
        print(f"Ejecutando {method} con {n_processes} procesos...")
        elapsed_time = execute_method(method, n_processes, matrix_file)
        results[method][n_processes] = elapsed_time
        print(f"  Tiempo: {elapsed_time:.4f} segundos")

# Mostrar resultados
print("\nResultados de tiempos:")
for method, timings in results.items():
    print(f"\nMétodo: {method}")
    for n_processes, elapsed_time in timings.items():
        print(f"  {n_processes} procesos: {elapsed_time:.4f} segundos")