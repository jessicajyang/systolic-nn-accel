#!/usr/bin/env python3
"""Synthesize sa_accel with Yosys for a generic Xilinx 7-series target and report resource usage.

Writes docs/synthesis.md. Usage: python scripts/synth.py [--sizes 4 8 16]

This reports AREA (LUTs, flip-flops, DSP slices, RAM primitives). Fmax needs
place-and-route timing analysis, which is done in Vivado.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ["rtl/pe.v", "rtl/systolic_array.v", "rtl/sa_accel.v"]


def parse_cells(text):
    """Return {cell_type: count} from the last 'Number of cells' block of a Yosys `stat` report."""
    block = text.rsplit("Number of cells", 1)[-1]
    cells = {}
    for line in block.splitlines():
        m = re.match(r"^\s+(\$?\w+)\s+(\d+)\s*$", line)
        if m:
            cells[m.group(1)] = int(m.group(2))
    return cells


def summarize(cells):
    lut = sum(v for k, v in cells.items() if k.startswith("LUT"))
    ff = sum(v for k, v in cells.items() if k.startswith("FD"))
    dsp = sum(v for k, v in cells.items() if k.startswith("DSP"))
    ram = sum(v for k, v in cells.items() if k.startswith("RAM") or k.startswith("RAMB"))
    return lut, ff, dsp, ram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    if not shutil.which("yosys"):
        sys.exit("Yosys not found. Install it: sudo apt install yosys  (or: brew install yosys)")

    rows, raw = [], []
    for n in args.sizes:
        script = (f"read_verilog {' '.join(SRC)}; chparam -set N {n} sa_accel; "
                  f"synth_xilinx -top sa_accel -flatten; tee -o /dev/stdout stat")
        out = subprocess.run(["yosys", "-q", "-p", script], cwd=ROOT,
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout + out.stderr)
            sys.exit(f"Yosys failed for N={n}")
        cells = parse_cells(out.stdout)
        if not cells:
            print(out.stdout)
            sys.exit(f"Could not parse Yosys stat output for N={n}")
        lut, ff, dsp, ram = summarize(cells)
        rows.append((n, lut, ff, dsp, ram))
        raw.append((n, cells))

    table = ("| N | LUTs | Flip-flops | DSP48 | RAM primitives |\n|---|---|---|---|---|\n"
             + "".join(f"| {n} | {l:,} | {f:,} | {d} | {r} |\n" for n, l, f, d, r in rows))
    detail = "".join(f"\n<details><summary>N={n} raw cell counts</summary>\n\n```\n"
                     + "\n".join(f"{k:<14}{v}" for k, v in sorted(c.items()))
                     + "\n```\n</details>\n" for n, c in raw)
    (ROOT / "docs" / "synthesis.md").write_text(
        "Yosys `synth_xilinx` (generic 7-series), KMAX = 64, int8 x int8 -> int32.\n\n"
        + table + detail)
    print(table)


if __name__ == "__main__":
    main()
