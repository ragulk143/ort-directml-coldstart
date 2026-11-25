# Where Does the Time Go? Cold-Start Pipeline Compilation vs. Steady-State Latency of Vision Models Across CPU, Integrated GPU, and Discrete GPU Execution Providers

Reproducible C++17 / ONNX Runtime DirectML benchmark harness, raw experimental telemetry, and statistical analysis scripts.

## Hardware & Software Environment
- **Host System:** Infinix GT Book (Intel Core i5-12450H, 8 cores / 12 threads, Windows 11 AC Best Performance)
- **Discrete GPU (DML Device 0):** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver `32.0.15.8142`)
- **Integrated GPU (DML Device 1):** Intel(R) UHD Graphics (Driver `31.0.101.4887`)
- **Runtime:** ONNX Runtime C API `v1.20.1` with DirectML Execution Provider
- **Compiler:** MinGW-W64 `g++ 16.2.0` (`-O3 -std=c++17`)

## Repository Structure
- `src/bench.cpp` — Isolated child-process cold-start and steady-state latency harness
- `src/hybrid.cpp` — Asynchronous dual-session CPU-to-GPU warm-up dispatcher
- `scripts/run_matrix.ps1` — Randomized 160-trial experimental matrix runner
- `results/coldstart_raw.csv` — Raw telemetry across 160 baseline trials
- `results/hybrid_raw.csv` — Raw telemetry across 40 hybrid dispatcher trials
- `analysis/analyze.py` — Non-parametric statistical analysis (median, IQR, p95, Mann-Whitney U) and vector PDF figure generator
- `paper/main.tex` & `paper/refs.bib` — Springer Nature (`sn-jnl`) manuscript source

## How to Reproduce
1. Download ONNX Runtime DirectML (`v1.20.1`) and extract headers/DLLs to `ort/`.
2. Download `squeezenet1.1-7.onnx` and `mobilenetv2-12.onnx` into `models/`.
3. Compile both C++ binaries:
   ```powershell
   g++ -O3 -std=c++17 -D_stdcall=__stdcall src\bench.cpp -I ort\build\native\include -L . -lonnxruntime -o bench.exe
   g++ -O3 -std=c++17 -D_stdcall=__stdcall src\hybrid.cpp -I ort\build\native\include -L . -lonnxruntime -o hybrid.exe
   ```
4. Execute `.\scripts\run_matrix.ps1` and `python analysis\analyze.py`.
