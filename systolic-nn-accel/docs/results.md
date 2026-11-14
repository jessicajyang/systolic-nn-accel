Workload: int8 MLP 64-32-10, 450 test images, all matrix multiplies on the accelerator model. Baseline: idealised scalar core, 1 MAC/cycle.

| N | Peak MACs/cycle | Tiles | Compute cycles | Utilization | Speedup (compute) | Speedup (incl. bus) |
|---|---|---|---|---|---|---|
| 4 | 16 | 1243 | 76,162 | 87.4% | 14.0x | 1.6x |
| 8 | 64 | 342 | 23,028 | 72.3% | 46.3x | 3.2x |
| 16 | 256 | 87 | 7,250 | 57.4% | 147.0x | 6.0x |
| 32 | 1024 | 30 | 3,300 | 31.5% | 322.9x | 8.4x |
