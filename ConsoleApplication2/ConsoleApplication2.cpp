#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include </usr/local/cuda/include/nvtx3/nvToolsExt.h>


int main(int argc, char** argv) {

    int size = 512;
    double accuracy = 10e-6;
    int iterations = 1000000;
    for (int i = 0; i < argc - 1; i++) {
        std::string arg = argv[i];
        if (arg == "-accuracy") {
            std::string dump = std::string(argv[i + 1]);
            accuracy = std::stod(dump);
        }
        else if (arg == "-a") {
            size = std::stoi(argv[i + 1]);
        }
        else if (arg == "-i") {
            iterations = std::stoi(argv[i + 1]);
        }
    }
    double* A = new double[size * size];
    double* Anew = new double[size * size];
    std::memset(A, 0, sizeof(double) * size * size);
    int grid_size = size * size;
    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * size - 1] = 30.0;
    A[size * (size - 1)] = 20.0;
    double step = (20.0 - 10.0) / (size - 1);
    clock_t start = clock();
    {
        for (int i = 1; i < size - 1; i++) {
            A[i] = 10.0 + i * step;
            A[i * size + (size - 1)] = 20.0 + i * step;
            A[i * size] = 10.0 + i * step;
            A[size * (size - 1) + i] = 20.0 + i * step;
        }
    }
    std::memcpy(Anew, A, sizeof(double) * grid_size);
    clock_t end = clock();
    double work_time = double(end - start) / CLOCKS_PER_SEC;
    double err = 1.0;
    int iter = 0;
    double tol = accuracy;
    int eter_max = iterations;
    start = clock();
nvtxRangePushA("mainloop");
#pragma acc enter data copyin(Anew[0:grid_size], A[0:grid_size], err, iter, tol, eter_max)
    {
    while (err > tol && iter < eter_max) {
        iter++;
        if (iter % 100 == 0) {
#pragma acc kernels async(1)
            err = 0.0;
#pragma acc update device(err) async(1)
        }
#pragma acc data present(A, Anew, err)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:err) async(1)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                err = fmax(err, Anew[i * size + j] - A[i * size + j]);
            }
        }

        if (iter % 100 == 0) {
#pragma acc update host(err) async(1)
#pragma acc wait(1)
        }

        double* tmp = A;
        A = Anew;
        Anew = tmp;
    }
    }
nvtxRangePop();
    end = clock();

    work_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "~Wна~Gение о~Hибки: " << err << std::endl;
    std::cout << "~Zол-во и~B~Bе~@а~Fий: " << iter << std::endl;
    free(A);
    free(Anew);
    return 0;
}
