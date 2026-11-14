#!/usr/bin/env python3
"""Run the Verilog testbench over many (N, K, data) cases with Icarus Verilog.

For each array size N the DUT is compiled once; then for each case this script
writes random int8 operands + the Python reference result to build/*.hex, runs
the simulation, and requires a PASS line.

Usage: python scripts/run_rtl_tests.py [--sizes 2 4 8] [--vcd]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
KMAX = 64


def write_hex(path, arr, width):
    digits = width // 4
    with open(path, "w") as f:
        for v in np.asarray(arr).flatten():
            f.write(f"{int(v) & ((1 << width) - 1):0{digits}x}\n")


def cases(n, rng):
    ks = sorted({1, 2, n, n + 1, 2 * n, KMAX})
    ks += [int(x) for x in rng.integers(1, KMAX + 1, size=4)]
    for k in ks:
        yield f"random K={k}", k, rng.integers(-128, 128, (n, k)), rng.integers(-128, 128, (k, n))
    yield "all -128", KMAX, np.full((n, KMAX), -128), np.full((KMAX, n), -128)
    yield "all +127", KMAX, np.full((n, KMAX), 127), np.full((KMAX, n), 127)
    yield "zeros", n, np.zeros((n, n), int), np.zeros((n, n), int)
    yield "identity", n, np.eye(n, dtype=int) * 7, rng.integers(-128, 128, (n, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--vcd", action="store_true", help="dump build/wave.vcd (last case)")
    args = ap.parse_args()

    if not shutil.which("iverilog") or not shutil.which("vvp"):
        sys.exit("Icarus Verilog not found. Install it: sudo apt install iverilog  (or: brew install icarus-verilog)")

    BUILD.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    total = failed = 0

    for n in args.sizes:
        sim = BUILD / f"sim_n{n}.vvp"
        subprocess.run(
            ["iverilog", "-g2005", "-Wall", f"-Ptb_sa_accel.N={n}", f"-Ptb_sa_accel.KMAX={KMAX}",
             "-o", str(sim), "tb/tb_sa_accel.v", "rtl/sa_accel.v", "rtl/systolic_array.v", "rtl/pe.v"],
            cwd=ROOT, check=True)
        for name, k, a, b in cases(n, rng):
            c = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
            write_hex(BUILD / "a.hex", a, 8)
            write_hex(BUILD / "b.hex", b, 8)
            write_hex(BUILD / "c.hex", c, 32)
            cmd = ["vvp", str(sim), f"+K={k}"] + (["+vcd"] if args.vcd else [])
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
            ok = "PASS" in out and "FAIL" not in out
            total += 1
            failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  N={n:<2} {name}")
            if not ok:
                print(out)

    print(f"\n{total - failed}/{total} RTL tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
