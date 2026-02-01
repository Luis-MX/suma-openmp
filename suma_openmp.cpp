#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>   // std::strtoll, std::strtol
#include <cmath>     // std::fabs

#ifdef _OPENMP
  #include <omp.h>
#endif

// Imprime los primeros "mostrar" elementos de un arreglo (para verificación rápida).
static void imprimeArreglo(const std::vector<double>& arr,
                           const std::string& nombre,
                           int mostrar) {
    std::cout << nombre << ": ";
    const int m = std::min<int>(mostrar, static_cast<int>(arr.size()));
    for (int i = 0; i < m; ++i) {
        std::cout << std::fixed << std::setprecision(1) << arr[i];
        if (i + 1 < m) std::cout << ", ";
    }
    std::cout << "\n";
}

// Verifica que c[i] = a[i] + b[i] para todo i, con tolerancia numérica.
static bool validaSuma(const std::vector<double>& a,
                       const std::vector<double>& b,
                       const std::vector<double>& c) {
    const double eps = 1e-9;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        const double esperado = a[i] + b[i];
        if (std::fabs(c[i] - esperado) > eps) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // Parámetros con valores por defecto (pueden sobrescribirse por CLI):
    // N: tamaño de arreglos
    // chunk: tamaño de bloque para schedule(static, chunk)
    // mostrar: cuántos elementos imprimir
    // reps: repeticiones para promediar tiempo (reduce ruido)
    long long N = 1'000'000;
    int chunk = 1'000;
    int mostrar = 20;
    int reps = 5;

    if (argc >= 2) N = std::strtoll(argv[1], nullptr, 10);
    if (argc >= 3) chunk = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    if (argc >= 4) mostrar = static_cast<int>(std::strtol(argv[3], nullptr, 10));
    if (argc >= 5) reps = static_cast<int>(std::strtol(argv[4], nullptr, 10));

    if (N <= 0) {
        std::cerr << "Error: N debe ser > 0.\n";
        return 1;
    }
    if (chunk <= 0) chunk = 1;
    if (mostrar < 0) mostrar = 0;
    if (reps <= 0) reps = 1;

    std::cout << "Suma de arreglos con OpenMP\n";
    std::cout << "N=" << N << ", chunk=" << chunk << ", mostrar=" << mostrar << ", reps=" << reps << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP: habilitado (max threads = " << omp_get_max_threads() << ")\n";
#else
    std::cout << "OpenMP: no detectado (se ejecuta secuencial)\n";
#endif
    std::cout << "\n";

    // Arreglos de entrada (A, B) y dos salidas:
    // - c_seq: referencia secuencial
    // - c_par: resultado paralelo
    std::vector<double> a(static_cast<size_t>(N));
    std::vector<double> b(static_cast<size_t>(N));
    std::vector<double> c_seq(static_cast<size_t>(N));
    std::vector<double> c_par(static_cast<size_t>(N));

    // Inicialización determinística para facilitar verificación y reproducibilidad.
    for (long long i = 0; i < N; ++i) {
        a[static_cast<size_t>(i)] = static_cast<double>(i) * 10.0;
        b[static_cast<size_t>(i)] = static_cast<double>(i + 3) * 3.7;
        c_seq[static_cast<size_t>(i)] = 0.0;
        c_par[static_cast<size_t>(i)] = 0.0;
    }

    // Medición: se promedian tiempos sobre varias repeticiones.
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms  = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    double t_seq_ms = 0.0;
    double t_par_ms = 0.0;

    // Secuencial (referencia)
    for (int r = 0; r < reps; ++r) {
        auto t0 = now();
        for (long long i = 0; i < N; ++i) {
            const size_t idx = static_cast<size_t>(i);
            c_seq[idx] = a[idx] + b[idx];
        }
        auto t1 = now();
        t_seq_ms += ms(t0, t1);
    }
    t_seq_ms /= reps;

    // Paralelo (OpenMP): el ciclo no tiene dependencias entre iteraciones,
    // porque cada iteración escribe en una posición distinta (c_par[i]).
#ifdef _OPENMP
    for (int r = 0; r < reps; ++r) {
        auto t0 = now();
        #pragma omp parallel for default(none) shared(a, b, c_par, N, chunk) schedule(static, chunk)
        for (long long i = 0; i < N; ++i) {
            const size_t idx = static_cast<size_t>(i);
            c_par[idx] = a[idx] + b[idx];
        }
        auto t1 = now();
        t_par_ms += ms(t0, t1);
    }
    t_par_ms /= reps;
#else
    // Si OpenMP no está activo, se replica el cálculo secuencial en c_par.
    for (int r = 0; r < reps; ++r) {
        auto t0 = now();
        for (long long i = 0; i < N; ++i) {
            const size_t idx = static_cast<size_t>(i);
            c_par[idx] = a[idx] + b[idx];
        }
        auto t1 = now();
        t_par_ms += ms(t0, t1);
    }
    t_par_ms /= reps;
#endif

    // Verificación: se confirma que el resultado paralelo es correcto.
    const bool correcto = validaSuma(a, b, c_par);

    std::cout << "Tiempo promedio secuencial (ms): " << std::fixed << std::setprecision(3) << t_seq_ms << "\n";
    std::cout << "Tiempo promedio paralelo   (ms): " << std::fixed << std::setprecision(3) << t_par_ms << "\n";

    if (t_par_ms > 0.0) {
        const double speedup = t_seq_ms / t_par_ms;
        std::cout << "Speedup (seq/par): " << std::fixed << std::setprecision(3) << speedup << "x\n";
    }

    std::cout << "\nVerificación (primeros " << mostrar << " elementos)\n";
    imprimeArreglo(a, "A", mostrar);
    imprimeArreglo(b, "B", mostrar);
    imprimeArreglo(c_par, "C", mostrar);

    std::cout << "\nResultado de validación: " << (correcto ? "CORRECTO" : "ERROR") << "\n";
    return correcto ? 0 : 2;
}