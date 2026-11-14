"""Sweep array size N on the quantized-MLP workload and write docs/results.md + docs/speedup.png.

Run: python sw/benchmark.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_model import Accelerator
from quant_mlp import build, forward_int8, quantize_inputs

DOCS = Path(__file__).resolve().parent.parent / "docs"
SIZES = [4, 8, 16, 32]


def main():
    m = build()
    xq = quantize_inputs(m, m["xte"])
    ref = forward_int8(m, xq, lambda a, b: a @ b)

    rows = []
    for n in SIZES:
        acc = Accelerator(n=n)
        out = forward_int8(m, xq, acc.matmul)
        assert np.array_equal(out, ref), f"N={n}: accelerator output differs from NumPy reference"
        s = acc.stats
        rows.append((n, s.tiles, s.compute_cycles, s.utilization,
                     s.speedup_compute_only, s.speedup_end_to_end))

    header = ("| N | Peak MACs/cycle | Tiles | Compute cycles | Utilization | "
              "Speedup (compute) | Speedup (incl. bus) |\n|---|---|---|---|---|---|---|\n")
    body = "".join(f"| {n} | {n * n} | {t} | {c:,} | {u * 100:.1f}% | {sc:.1f}x | {se:.1f}x |\n"
                   for n, t, c, u, sc, se in rows)
    table = header + body
    (DOCS / "results.md").write_text(
        f"Workload: int8 MLP 64-32-10, {len(m['yte'])} test images, "
        f"all matrix multiplies on the accelerator model. Baseline: idealised scalar core, 1 MAC/cycle.\n\n"
        + table)
    print(table)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[4] for r in rows], 0.4, label="compute only")
    ax.bar(x + 0.2, [r[5] for r in rows], 0.4, label="incl. register-bus transfers")
    ax.set_xticks(x, [f"{r[0]}x{r[0]}" for r in rows])
    ax.set_xlabel("systolic array size")
    ax.set_ylabel("speedup vs 1 MAC/cycle scalar core")
    ax.set_title("Digit-MLP inference speedup")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "speedup.png")


if __name__ == "__main__":
    main()
