from slepc4py import SLEPc
from petsc4py import PETSc
import psutil
import time

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

        # # Imprimir información de la matriz cargada
        # rank = PETSc.COMM_WORLD.getRank()
        # if rank == 0:
        #     print("Matriz cargada correctamente desde el archivo:")
        #     A.view()

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
    
    # Medir la memoria antes de crear el solucionador
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 * 1024)  # Convertir a MB

    
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A)
    eps.setType(SLEPc.EPS.Type.ARNOLDI)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setDimensions(nev=A.getSize()[0])
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
    eps.setTolerances(1e-6, 5000)

    start_time = time.time()
    eps.solve()
    end_time = time.time()

    # Medir la memoria después de resolver el problema
    end_mem = process.memory_info().rss / (1024 ** 2)  # Convertir a MB

    elapsed_time = end_time - start_time
    mem_usage = end_mem - start_mem

    rank = PETSc.COMM_WORLD.getRank()
    nconv = eps.getConverged()
    if rank == 0:
        print(f"Número de valores propios convergidos: {nconv}")
        print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
        print(f"Uso de memoria adicional: {mem_usage:.2f} MB")

        # Guardar resultados correspondientes a los valores y vectores propios en un archivo de texto
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
