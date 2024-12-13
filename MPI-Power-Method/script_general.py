import subprocess
import os

# Configuración de parámetros
num_threads = [1, 2, 4, 8, 16]
matrix_sizes = [32, 128, 512, 1024, 2048]
iterations = 1000  # Número de iteraciones por defecto
verbose = 1  # Nivel de verbosidad
save_file = "results"

# Ruta del script a ejecutar
script_path = "power_method.py"

# Verificar que el archivo exista
if not os.path.isfile(script_path):
    raise FileNotFoundError(f"El archivo {script_path} no se encuentra en el directorio actual.")

# Crear directorio para guardar resultados si no existe
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Ejecutar combinaciones
for threads in num_threads:
    for size in matrix_sizes:
        # Construir comando para ejecutar el script con mpiexec
        command = [
            "mpiexec",
            "--use-hwthread-cpus",
            "--oversubscribe",
            "-n", str(threads),
            "/usr/bin/python3", script_path,
            "--matrix", str(size),
            "--iter", str(iterations),
            "--numpy", "0",
            "--verbose", str(verbose),
            "--save", f"{results_dir}/{save_file}_threads{threads}_size{size}"
        ]

        print(f"Ejecutando: {' '.join(command)}")

        try:
            # Ejecutar el comando y capturar la salida
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # Imprimir la salida del comando
            print(result.stdout)

            # Guardar errores si los hubiera
            if result.stderr:
                with open(f"{results_dir}/errors.log", "a") as error_log:
                    error_log.write(f"Errores para {threads} hilos y matriz {size}x{size}:\n")
                    error_log.write(result.stderr + "\n")

        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar para {threads} hilos y matriz {size}x{size}.")
            print(e.stderr)
            with open(f"{results_dir}/errors.log", "a") as error_log:
                error_log.write(f"Falló para {threads} hilos y matriz {size}x{size} con el error:\n")
                error_log.write(e.stderr + "\n")

print("Ejecuciones completas. Resultados guardados en el directorio 'results'.")
