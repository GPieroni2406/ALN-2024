#include <omp.h>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <iostream>

template<typename T = double>
struct Matrix {
    explicit Matrix(int n, T init = 0) : n(n), v(n * n, init) {}

    int n;
    std::vector<T> v;
};

template<typename T>
Matrix<T> cargar_matriz(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Fallo al abrir el archivo");

    int m, n, n_values;
    file >> m >> n >> n_values;
    assert(m == n);

    Matrix<T> M(n);

    for (int k = 0; k < n_values; k++) {
        int i, j;
        T v;
        file >> i >> j >> v;
        i -= 1;
        j -= 1;
        assert(0 <= i && i < n);
        assert(0 <= j && j < n);
        M.v[i * n + j] = v;
    }

    return std::move(M);
}

template<typename T>
void imprimir_matriz(const std::string &nombre, const Matrix<T> &M, int width = 8, int precision = 5) {
    std::cout << std::setprecision(precision) << "Matriz " << nombre << std::endl;
    for (int i = 0; i < M.n; i++) {
        for (int j = 0; j < M.n; j++) {
            std::cout << std::setw(width) << M.v[i * M.n + j] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void descomposicion_qr(const Matrix<T> &A,Matrix<T> &Q,Matrix<T> &R) {
    assert(A.n == Q.n);
    auto n = A.n;

    Q = A;

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            R.v[i * n + k] = 0;

    for (int i = 0; i < n; i++)
        R.v[i * n + i] = 1;

    for (int i = 0; i < n; i++) {

        T norm = 0;
        for (int k = 0; k < n; k++)
            norm += Q.v[i * n + k] * Q.v[i * n + k];
        norm = std::sqrt(norm);
        for (int k = 0; k < n; k++)
            Q.v[i * n + k] /= norm;
        R.v[i * n + i] *= norm;


#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            T q_i_dot_q_j = 0;
            for (int k = 0; k < n; k++)
                q_i_dot_q_j += Q.v[i * n + k] * Q.v[j * n + k];
            for (int k = 0; k < n; k++)
                Q.v[j * n + k] -= q_i_dot_q_j * Q.v[i * n + k];
            R.v[i * n + j] += q_i_dot_q_j;
        }
    }
}

template<typename T>
void multiplicar(const Matrix<T> &A,const Matrix<T> &B,Matrix<T> &C) {
    assert(A.n == B.n);
    assert(A.n == C.n);
    auto n = A.n;

#pragma omp parallel for
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            T sum = 0;
            for (int k = 0; k < n; k++)
                sum += A.v[i * n + k] * B.v[j * n + k];

            C.v[j * n + i] = sum;
        }
    }
}

template<typename T>
std::vector<T> algoritmo_qr(Matrix<T> &A, int iteraciones = 10) {
    const T tolerancia = 1e-6;

    Matrix<T> X = A; 
    Matrix<T> Q(A.n);
    Matrix<T> R(A.n); 
    std::vector<T> valores_propios(X.n, 0); // Inicializar valores propios

    for (int k = 0; k < iteraciones; k++) {
        descomposicion_qr(X, Q, R);  // Descomposición QR
        multiplicar(R, Q, X);          // Multiplicación R*Q

        // Verificar convergencia
        bool convergencia = true;
        for (int i = 0; i < X.n; i++) {
            T nuevo_valor = X.v[i * X.n + i];  // Valor propio actual
            if (std::abs(nuevo_valor - valores_propios[i]) > tolerancia) {
                convergencia = false;
            }
            valores_propios[i] = nuevo_valor;  // Actualizar valores propios
        }

        if (convergencia) {
            std::cout << "convergencia despues de " << k + 1 << " iteraciones." << std::endl;
            break;
        }
    }

    return valores_propios;
}


int main(int argc, const char *const *argv) {
    if (argc != 5) {
        std::cerr << "Problema con los argumentos!";
        return -1;
    }

    using T = double;
    using clock = std::chrono::steady_clock;
    using ns = std::chrono::nanoseconds;

    int muestras = std::atoi(argv[1]);
    int cont_hilos = std::atoi(argv[2]);
    int iteraciones = std::atoi(argv[3]);
    std::string matrix_path = argv[4];
    std::vector<double> times;

    omp_set_num_threads(cont_hilos);

    Matrix<T> A = cargar_matriz<T>(matrix_path); // Col mayor

    for (int i = 0; i < muestras; i++) {
        auto tiempo_act = clock::now();
        algoritmo_qr(A, iteraciones);
        times.push_back(static_cast<double >(std::chrono::duration_cast<ns>(clock::now() - tiempo_act).count()) / 1e9f);
        std::cout << i << " ";
    }

    double promedio = std::reduce(times.begin(), times.end(), 0.0) / static_cast<double>(muestras);
    double tiempo_min = *std::min_element(times.begin(), times.end());
    double tiempo_max = *std::max_element(times.begin(), times.end());
    double sd = std::transform_reduce(times.begin(), times.end(), 0.0, std::plus<>(),
                                      [=](auto x) { return (x - promedio) * (x - promedio); })
                / static_cast<double>(muestras - 1);
    std::cout << "resultado (promedio, tiempo min, tiempo max): "
              << promedio << ", "
              << tiempo_min << ", "
              << tiempo_max << ", "
              << sd << " (en segundos)" << std::endl;

    return 0;
}