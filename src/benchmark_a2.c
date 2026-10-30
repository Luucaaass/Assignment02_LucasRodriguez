#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include <sys/resource.h>
#endif

#define SEED 42
#define WARMUP_RUNS 2
#define MEASURED_RUNS 5
#define BLOCK_SIZE 64
#define TOLERANCE 1e-9

static double get_time_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static double get_peak_rss_mb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return (double)usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)usage.ru_maxrss / 1024.0;
#endif
#endif
}

static uint64_t rng_state = SEED;
static void set_seed(uint64_t s) { rng_state = s ? s : 1; }
static double rand_uniform(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return ((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

static void generate_matrix(double *M, int n, double density, uint64_t seed) {
    set_seed(seed);
    for (long i = 0; i < (long)n * n; i++) {
        double p = rand_uniform();
        if (p < density) {
            M[i] = rand_uniform() * 2.0 - 1.0;
            if (fabs(M[i]) < 1e-12) M[i] = 0.5;
        } else {
            M[i] = 0.0;
        }
    }
}

static double compute_checksum(const double *C, int n) {
    double sum = 0.0;
    for (long i = 0; i < (long)n * n; i += (n > 64 ? 17 : 1)) {
        sum += C[i];
    }
    return sum;
}


void dense_ijk(const double *A, const double *B, double *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[(long)i * n + k] * B[(long)k * n + j];
            }
            C[(long)i * n + j] = sum;
        }
    }
}

void dense_ikj(const double *A, const double *B, double *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[(long)i * n + k];
            const double *b_row = &B[(long)k * n];
            double *c_row = &C[(long)i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

void dense_tiled(const double *A, const double *B, double *C, int n, int bsize) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += bsize) {
        int i_max = (ii + bsize < n) ? ii + bsize : n;
        for (int kk = 0; kk < n; kk += bsize) {
            int k_max = (kk + bsize < n) ? kk + bsize : n;
            for (int jj = 0; jj < n; jj += bsize) {
                int j_max = (jj + bsize < n) ? jj + bsize : n;
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double a_ik = A[(long)i * n + k];
                        const double *b_row = &B[(long)k * n];
                        double *c_row = &C[(long)i * n];
                        for (int j = jj; j < j_max; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}


typedef struct {
    int n;
    long nnz;
    double *values;
    int *col_idx;
    int *row_ptr;
} CSRMatrix;

CSRMatrix* dense_to_csr(const double *M, int n) {
    CSRMatrix *csr = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    csr->n = n;
    long nnz = 0;
    for (long i = 0; i < (long)n * n; i++) {
        if (M[i] != 0.0) nnz++;
    }
    csr->nnz = nnz;
    csr->values = (double*)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(double));
    csr->col_idx = (int*)malloc((size_t)(nnz > 0 ? nnz : 1) * sizeof(int));
    csr->row_ptr = (int*)malloc((size_t)(n + 1) * sizeof(int));

    long idx = 0;
    csr->row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = M[(long)i * n + j];
            if (val != 0.0) {
                csr->values[idx] = val;
                csr->col_idx[idx] = j;
                idx++;
            }
        }
        csr->row_ptr[i + 1] = (int)idx;
    }
    return csr;
}

void free_csr(CSRMatrix *csr) {
    if (!csr) return;
    free(csr->values);
    free(csr->col_idx);
    free(csr->row_ptr);
    free(csr);
}

long sparse_csr_multiply(const CSRMatrix *A, const CSRMatrix *B, double *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *c_row = &C[(long)i * n];
        for (int pA = A->row_ptr[i]; pA < A->row_ptr[i + 1]; pA++) {
            int k = A->col_idx[pA];
            double a_val = A->values[pA];
            for (int pB = B->row_ptr[k]; pB < B->row_ptr[k + 1]; pB++) {
                int j = B->col_idx[pB];
                c_row[j] += a_val * B->values[pB];
            }
        }
    }
    long nnz_C = 0;
    for (long i = 0; i < (long)n * n; i++) {
        if (fabs(C[i]) > 1e-15) nnz_C++;
    }
    return nnz_C;
}


int verify_matrices(const double *Ref, const double *Test, int n, const char *label) {
    for (long i = 0; i < (long)n * n; i++) {
        double diff = fabs(Ref[i] - Test[i]);
        double denom = fmax(1.0, fabs(Ref[i]));
        if (diff > TOLERANCE && (diff / denom) > TOLERANCE) {
            fprintf(stderr, "[ERROR] Validation failed on %s (idx %ld: ref=%g, test=%g)\n",
                    label, i, Ref[i], Test[i]);
            return 0;
        }
    }
    return 1;
}

int run_correctness_suite(void) {
    printf("=== Running Correctness Validation Suite ===\n");
    int n = 67;
    double *A = malloc((size_t)n * n * sizeof(double));
    double *B = malloc((size_t)n * n * sizeof(double));
    double *I = calloc((size_t)n * n, sizeof(double));
    double *Z = calloc((size_t)n * n, sizeof(double));
    double *C_ref = malloc((size_t)n * n * sizeof(double));
    double *C_test = malloc((size_t)n * n * sizeof(double));

    for (int i = 0; i < n; i++) I[i * n + i] = 1.0;
    generate_matrix(A, n, 0.35, 101);
    generate_matrix(B, n, 0.35, 202);

    dense_ijk(A, I, C_test, n);
    if (!verify_matrices(A, C_test, n, "Identity (A * I == A)")) return 0;
    dense_ijk(A, Z, C_test, n);
    if (!verify_matrices(Z, C_test, n, "Zero (A * 0 == 0)")) return 0;

    dense_ijk(A, B, C_ref, n);
    dense_ikj(A, B, C_test, n);
    if (!verify_matrices(C_ref, C_test, n, "dense_ikj vs dense_ijk")) return 0;
    dense_tiled(A, B, C_test, n, BLOCK_SIZE);
    if (!verify_matrices(C_ref, C_test, n, "dense_tiled vs dense_ijk")) return 0;

    CSRMatrix *csrA = dense_to_csr(A, n);
    CSRMatrix *csrB = dense_to_csr(B, n);
    sparse_csr_multiply(csrA, csrB, C_test, n);
    if (!verify_matrices(C_ref, C_test, n, "sparse_csr vs dense_ijk")) return 0;

    free_csr(csrA); free_csr(csrB);
    free(A); free(B); free(I); free(Z); free(C_ref); free(C_test);
    printf("[OK] All correctness checks passed (tol=%g).\n\n", TOLERANCE);
    return 1;
}

int main(void) {
    if (!run_correctness_suite()) return EXIT_FAILURE;

    FILE *fpA = fopen("results/raw_partA_dense.csv", "w");
    if (!fpA) { perror("Cannot open results/raw_partA_dense.csv"); return EXIT_FAILURE; }
    fprintf(fpA, "n,method,numeric_type,rep,kernel_time_s,structure_mem_mb,peak_rss_mb,checksum\n");

    int sizes_A[] = {128, 256, 512, 768, 1024};
    int num_sizes_A = sizeof(sizes_A) / sizeof(sizes_A[0]);
    const char *methods_A[] = {"dense_ijk", "dense_ikj", "dense_tiled"};

    printf("=== Running Part A: Dense Optimisation ===\n");
    for (int s = 0; s < num_sizes_A; s++) {
        int n = sizes_A[s];
        double *A = malloc((size_t)n * n * sizeof(double));
        double *B = malloc((size_t)n * n * sizeof(double));
        double *C = malloc((size_t)n * n * sizeof(double));
        generate_matrix(A, n, 1.0, SEED + n);
        generate_matrix(B, n, 1.0, SEED + n + 1000);
        double struct_mem_mb = (3.0 * (double)n * n * sizeof(double)) / (1024.0 * 1024.0);

        for (int m = 0; m < 3; m++) {
            for (int r = -WARMUP_RUNS; r < MEASURED_RUNS; r++) {
                double t0 = get_time_sec();
                if (m == 0) dense_ijk(A, B, C, n);
                else if (m == 1) dense_ikj(A, B, C, n);
                else dense_tiled(A, B, C, n, BLOCK_SIZE);
                double elapsed = get_time_sec() - t0;
                double chk = compute_checksum(C, n);

                if (r >= 0) {
                    fprintf(fpA, "%d,%s,float64,%d,%.9f,%.4f,%.4f,%.6f\n",
                            n, methods_A[m], r + 1, elapsed, struct_mem_mb, get_peak_rss_mb(), chk);
                    fflush(fpA);
                }
            }
            printf("  [Part A] n=%4d | %-12s done\n", n, methods_A[m]);
        }
        free(A); free(B); free(C);
    }
    fclose(fpA);

    FILE *fpB = fopen("results/raw_partB_sparse.csv", "w");
    if (!fpB) { perror("Cannot open results/raw_partB_sparse.csv"); return EXIT_FAILURE; }
    fprintf(fpB, "n,density,sparsity,method,format,rep,conversion_time_s,kernel_time_s,total_time_s,nnz_A,nnz_B,nnz_C,structure_mem_mb,peak_rss_mb,checksum\n");

    int sizes_B[] = {512, 1024};
    int num_sizes_B = sizeof(sizes_B) / sizeof(sizes_B[0]);
    double densities[] = {0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50};
    int num_densities = sizeof(densities) / sizeof(densities[0]);

    printf("\n=== Running Part B: Sparse (CSR) vs Dense ===\n");
    for (int s = 0; s < num_sizes_B; s++) {
        int n = sizes_B[s];
        double *A = malloc((size_t)n * n * sizeof(double));
        double *B = malloc((size_t)n * n * sizeof(double));
        double *C = malloc((size_t)n * n * sizeof(double));

        for (int d = 0; d < num_densities; d++) {
            double density = densities[d];
            double sparsity = 1.0 - density;
            generate_matrix(A, n, density, SEED + n * 10 + d);
            generate_matrix(B, n, density, SEED + n * 20 + d);

            double dense_mem_mb = (3.0 * (double)n * n * sizeof(double)) / (1024.0 * 1024.0);

            for (int r = -WARMUP_RUNS; r < MEASURED_RUNS; r++) {
                double t0 = get_time_sec();
                dense_ikj(A, B, C, n);
                double k_time = get_time_sec() - t0;
                double chk = compute_checksum(C, n);
                if (r >= 0) {
                    fprintf(fpB, "%d,%.4f,%.4f,dense_ikj,dense_row_major,%d,0.000000000,%.9f,%.9f,%ld,%ld,%ld,%.4f,%.4f,%.6f\n",
                            n, density, sparsity, r + 1, k_time, k_time,
                            (long)n*n, (long)n*n, (long)n*n, dense_mem_mb, get_peak_rss_mb(), chk);
                    fflush(fpB);
                }
            }

            for (int r = -WARMUP_RUNS; r < MEASURED_RUNS; r++) {
                double tc0 = get_time_sec();
                CSRMatrix *csrA = dense_to_csr(A, n);
                CSRMatrix *csrB = dense_to_csr(B, n);
                double conv_time = get_time_sec() - tc0;

                double tk0 = get_time_sec();
                long nnz_C = sparse_csr_multiply(csrA, csrB, C, n);
                double k_time = get_time_sec() - tk0;
                double total_time = conv_time + k_time;
                double chk = compute_checksum(C, n);

                double csr_bytes = (double)(csrA->nnz + csrB->nnz) * (sizeof(double) + sizeof(int))
                                 + 2.0 * (n + 1) * sizeof(int)
                                 + (double)n * n * sizeof(double);
                double csr_mem_mb = csr_bytes / (1024.0 * 1024.0);

                if (r >= 0) {
                    fprintf(fpB, "%d,%.4f,%.4f,sparse_csr,CSR,%d,%.9f,%.9f,%.9f,%ld,%ld,%ld,%.4f,%.4f,%.6f\n",
                            n, density, sparsity, r + 1, conv_time, k_time, total_time,
                            csrA->nnz, csrB->nnz, nnz_C, csr_mem_mb, get_peak_rss_mb(), chk);
                    fflush(fpB);
                }
                free_csr(csrA);
                free_csr(csrB);
            }
            printf("  [Part B] n=%4d | density=%5.1f%% (sparsity=%5.1f%%) done\n",
                   n, density * 100.0, sparsity * 100.0);
        }
        free(A); free(B); free(C);
    }
    fclose(fpB);
    printf("\nAll benchmarks finished! CSV files saved in results/\n");
    return EXIT_SUCCESS;
}
