from petsc4py import PETSc
import numpy as np

def generar_matriz_dispersa_simetrica(n, densidad=0.01, nombre_archivo="matriz.dat"):

    # Crear matriz dispersa en formato PETSc
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.AIJ)  # Matriz dispersa
    A.setFromOptions()
    A.setUp()

    # Rellenar con valores simétricos aleatorios
    for i in range(n):
        for j in range(i, n):
            if np.random.rand() < densidad:  # Controlar la densidad
                valor = np.random.uniform(-10, 10)
                A[i, j] = valor
                A[j, i] = valor  # Hacerla simétrica

    A.assemble()

    # Guardar en archivo binario PETSc
    visor = PETSc.Viewer().createBinary(nombre_archivo, mode='w')
    A.view(visor)
    print(f"Matriz generada y guardada en {nombre_archivo} (tamaño: {n}x{n}, densidad: {densidad})")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # Tamaño por defecto
    nombre_archivo = sys.argv[2] if len(sys.argv) > 2 else "matriz.dat"
    generar_matriz_dispersa_simetrica(n, densidad=0.01, nombre_archivo=nombre_archivo)