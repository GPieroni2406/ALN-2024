from slepc4py import SLEPc
from petsc4py import PETSc

def create_sparse_hilbert_matrix(n):
    """
    Crea una matriz dispersa de Hilbert.
    """
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


def solve_krylov_schur(filename="matrix.dat", output_file="results/results_kyrlovschur.txt"):
    A = load_matrix(filename=filename)
    if A is None:
        print("Error: La matriz no se cargó correctamente. Verifica el archivo y vuelve a intentarlo.")
        exit(1)  # Salir si la matriz no es válida
    eps = SLEPc.EPS().create()
    eps.setOperators(A)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setTolerances(1e-6, 5000)

    # Opciones específicas para Krylov-Schur
    opts = PETSc.Options()
    opts["eps_krylovschur_restart"] = 0.7  # Establece un valor dentro del rango permitido [0.1, 0.9]
    opts["eps_krylovschur_locking"] = None  # Habilita el locking automáticamente

    # Leer opciones de la línea de comandos o configuraciones personalizadas
    eps.setFromOptions()

    # Resolver
    eps.solve()

    # Obtener y mostrar resultados
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

    # Liberar memoria
    A.destroy()
    eps.destroy()
solve_krylov_schur()