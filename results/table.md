| backend | variant | L | flips/ns | eff. GB/s | % of copy BW |
|---|---|---:|---:|---:|---:|
| NumPy | int8 | 256 | 0.06 | 0.2 | 0.1 |
| torch eager | int8 | 1024 | 0.16 | 0.5 | 0.1 |
| torch.compile | int8 | 1024 | 1.08 | 3.2 | 0.9 |
| Triton | int8 | 4096 | 35.42 | 106.2 | 30.0 |
| CUDA C (CuPy) | int8 | 1024 | 46.48 | 139.4 | 39.4 |
| Triton | msc | 8192 | 192.57 | 72.2 | 20.4 |
| CUDA C (CuPy) | msc | 8192 | 150.30 | 56.4 | 15.9 |
| CUDA C (CuPy) | msc-eager32 | 8192 | 76.74 | 28.8 | 8.1 |
