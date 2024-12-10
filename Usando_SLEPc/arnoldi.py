from slepc4py import SLEPc
from petsc4py import PETSc

def create_sparse_hilbert_matrix(n):
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
    A.assemble()
    return A

def load_matrix(filename="matrix.dat"):
    """
    Carga una matriz desde un archivo binario PETSc y verifica su contenido.
    """
    try:
        # Crear el objeto Mat
        A = PETSc.Mat().create()
        
        # Crear el viewer para leer la matriz desde el archivo binario
        viewer = PETSc.Viewer().createBinary(filename, mode='r')
        
        # Leer la matriz desde el archivo
        A.load(viewer)
        
        # Verificar el tipo de matriz
        if not A.getType():
            A.setType(PETSc.Mat.Type.AIJ)  # Configurar como matriz dispersa si no está definido
        
        A.setFromOptions()  # Configurar opciones automáticamente
        A.setUp()  # Configurar dimensiones y propiedades

        # Imprimir información de la matriz cargada
        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            print("Matriz cargada correctamente desde el archivo:")
            A.view()

        return A
    except PETSc.Error as e:
        print(f"Error de PETSc al cargar la matriz desde el archivo {filename}: {e}")
    except Exception as e:
        print(f"Error general al cargar la matriz desde el archivo {filename}: {e}")
    return None


def solve_arnoldi(filename="matrix.dat", output_file="results/results_arnoldi.txt"):
    A = load_matrix(filename=filename)
    if A is None:
        print("Error: La matriz no se cargó correctamente. Verifica el archivo y vuelve a intentarlo.")
        exit(1)  # Salir si la matriz no es válida
    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setType(SLEPc.EPS.Type.ARNOLDI)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setTolerances(1e-6, 5000)

    eps.solve()

    nconv = eps.getConverged()
    rank = PETSc.COMM_WORLD.getRank()
    if rank == 0:
        with open(output_file, "a") as f:
            f.write(f"Número de valores propios convergidos: {nconv}\n")
            if nconv > 0:
                for i in range(nconv):
                    eigenvalue = eps.getEigenvalue(i)
                    f.write(f"  λ[{i}] = {eigenvalue:.7f}\n")
                    #xr, _ = A.getVecs()
                    #eps.getEigenvector(i, xr)
                    #f.write(f"  Vector propio [{i}] = {xr[:].tolist()}\n")

    A.destroy()
    eps.destroy()

solve_arnoldi()
