from petsc4py import PETSc
import numpy as np

from petsc4py import PETSc

def generate_symmetric_sparse_matrix(n, density=0.01, filename="matrix.dat"):

    # Crear matriz dispersa en formato PETSc
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.AIJ)  # Matriz dispersa
    A.setFromOptions()
    A.setUp()

    # Rellenar con valores simétricos aleatorios
    for i in range(n):
        for j in range(i, n):
            if np.random.rand() < density:  # Controlar la densidad
                value = np.random.uniform(-10, 10)
                A[i, j] = value
                A[j, i] = value  # Hacerla simétrica

    A.assemble()

    # Guardar en archivo binario PETSc
    viewer = PETSc.Viewer().createBinary(filename, mode='w')
    #viewer(A)
    A.view(viewer)
    print(f"Matriz generada y guardada en {filename} (tamaño: {n}x{n}, densidad: {density})")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000  # Tamaño por defecto
    filename = sys.argv[2] if len(sys.argv) > 2 else "matrix.dat"
    generate_symmetric_sparse_matrix(n, density=0.01, filename=filename)