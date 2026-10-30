# Assignment 2: Optimised Matrix Multiplication Approaches and Sparse Matrices

**Author:** Lucas Rodríguez Hernández  
**Course:** Big Data — Grado en Ciencia e Ingeniería de Datos (ULPGC)  
**Academic Year:** 2026–2027  

## 1. Environment and System Specifications
- **Operating System:** Windows 11 (64-bit) / PowerShell
- **C Compiler:** GCC 11+ via MinGW-w64 (`gcc -O2 -std=c11 -Wall src/benchmark_a2.c -o benchmark_a2.exe -lm -lpsapi`)[cite: 1]
- **Python Interpreter:** CPython 3.10+ (with `pandas`, `numpy`, and `matplotlib` for analysis and plotting)[cite: 1]
- **Timer and Profiling:** Monotonic timers (`QueryPerformanceCounter` via Win32 API), Peak Resident Set Size tracking (`K32GetProcessMemoryInfo` via `psapi.h`), and deterministic 64-bit Xorshift PRNG (seed=42)

## 2. Repository Structure
```text
Assignment02_LucasRodriguez_Materials/
├── figures/                            # Generated evaluation figures
│   ├── fig1_dense_runtime.png          # Part A: Runtime vs. matrix dimension n (log scale)
│   ├── fig2_dense_speedup.png          # Part A: Speedup of dense optimisations vs. baseline
│   ├── fig3_sparse_vs_dense_breakeven.png # Part B: CSR vs. dense runtime across density levels
│   └── fig4_sparse_memory_and_fillin.png  # Part B: Structure memory footprint and output fill-in
├── results/                            # Raw experimental measurements and aggregated summaries
│   ├── raw_partA_dense.csv             # Part A raw repetitions (time, memory, checksum)
│   ├── raw_partB_sparse.csv            # Part B raw repetitions (conversion, kernel, nnz)
│   ├── summary_partA_dense.csv         # Part A aggregated statistics (median, IQR, speedup)
│   └── summary_partB_sparse.csv        # Part B aggregated statistics (median, IQR, fill-in)
├── scripts/                            # Data analysis and visualization scripts
│   └── analyze_and_plot.py             # Summary statistics (median, IQR) and plot generator
├── src/                                # Implementation source code
│   └── benchmark_a2.c                  # Unified benchmark suite (correctness, Part A and Part B)
├── .gitignore                          # Git exclusions (executables, binaries, __pycache__)
└── README.md                           # Build, execution, and reproduction guide
```
## 3. Reproduction Instructions

- **Step 1:** Compile the unified C benchmark suite with optimization level -O2 and PSAPI linking
```bash
gcc -O2 -std=c11 -Wall src/benchmark_a2.c -o benchmark_a2.exe -lm -lpsapi
```
- **Step 2:** Execute the correctness validation suite (Identity, Zero, cross-validation) and run benchmarks (2 warm-up runs, 5 measured repetitions per configuration)
```bash
./benchmark_a2.exe
```
- **Step 3:** Compute summary statistics (median, IQR, speedup, output density) and generate empirical plots
```bash
py scripts/analyze_and_plot.py
```
