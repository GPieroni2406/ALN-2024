import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Configuraciones de parámetros
    hilos = [16, 8, 4, 2, 1]
    muestras = [1]
    iteraciones = [1000]
    matrices = ["matriz_10.mtx", "matriz_100.mtx", "matriz_500.mtx", "matriz_1000.mtx"]

    for f in matrices:
            for s in muestras:  # Iterar sobre cada valor de muestras
                for i in iteraciones:  # Iterar sobre cada valor de iteraciones
                    for t in hilos:  # Iterar sobre cada valor de hilos
                        print(f"Exec muestras={s} hilos={t} iteraciones={i} matriz={f}")
                        # Ejecutar el comando con las rutas ajustadas
                        subprocess.check_call(["../algoritmo_qr", str(s), str(t), str(i), f"../data/{f}"])



if __name__ == '__main__':
    main()
