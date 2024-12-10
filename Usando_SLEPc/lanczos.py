from slepc4py import SLEPc
from petsc4py import PETSc

def create_sparse_hilbert_matrix(n):
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setType(PETSc.Mat.Type.AIJ)  # Formato disperso
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
        viewer = PETSc.Viewer().createBinary(filename, mode='r')  # Abrir el archivo como binario

        # Leer la matriz desde el archivo
        A.load(viewer)
        
        A.setFromOptions()  # Asegurar que las opciones se configuren automáticamente
        A.setUp()  # Configurar dimensiones y propiedades

        # Verificar y mostrar contenido
        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            print("Contenido de la matriz cargada:")
            A.view()

        return A
    except Exception as e:
        print(f"Error al cargar la matriz desde el archivo {filename}: {e}")
        return None


def solve_lanczos(filename="matrix.dat", output_file="results/results_lanczos.txt"):
    #A = create_sparse_hilbert_matrix(n)
    A = load_matrix(filename=filename)
    if A is None:
        print("Error: La matriz no se cargó correctamente. Verifica el archivo y vuelve a intentarlo.")
        exit(1)  # Salir si la matriz no es válida
    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setType(SLEPc.EPS.Type.LANCZOS)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setTolerances(1e-6, 5000)
    # Configuración específica para Lanczos
    opts = PETSc.Options()
    opts["eps_lanczos_reorthog"] = "local"  # Tipo de re-ortogonalización
    opts["eps_lanczos_restart"] = 20       # Máximo tamaño del subespacio antes del reinicio

    eps.setFromOptions()
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
solve_lanczos()