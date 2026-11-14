"""Digit-recognition demo: int8 MLP inference with every matrix multiply on the accelerator model.

Run: python sw/demo.py [--n 8]
"""
import argparse

import numpy as np

from accel_model import Accelerator
from quant_mlp import build, forward_int8, quantize_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="systolic array size N")
    args = ap.parse_args()

    m = build()
    xq = quantize_inputs(m, m["xte"])
    yte = m["yte"]

    ref_logits = forward_int8(m, xq, lambda a, b: a @ b)            # plain NumPy int32
    acc = Accelerator(n=args.n)
    hw_logits = forward_int8(m, xq, acc.matmul)                     # on the accelerator model

    ref_acc = float(np.mean(ref_logits.argmax(1) == yte))
    hw_acc = float(np.mean(hw_logits.argmax(1) == yte))
    s = acc.stats

    print(f"Test images:                   {len(yte)}")
    print(f"float32 MLP accuracy:          {m['float_acc'] * 100:.2f}%")
    print(f"int8 NumPy accuracy:           {ref_acc * 100:.2f}%")
    print(f"int8 accelerator accuracy:     {hw_acc * 100:.2f}%")
    print(f"bit-exact vs NumPy int8:       {np.array_equal(ref_logits, hw_logits)}")
    print()
    print(f"Array size:                    {args.n} x {args.n}  ({args.n ** 2} MACs/cycle peak)")
    print(f"Tiles executed:                {s.tiles}")
    print(f"Accelerator compute cycles:    {s.compute_cycles:,}")
    print(f"Scalar baseline (1 MAC/cycle): {s.baseline_cycles:,}")
    print(f"Array utilization:             {s.utilization * 100:.1f}%")
    print(f"Speedup (compute only):        {s.speedup_compute_only:.1f}x")
    print(f"Speedup (incl. bus transfers): {s.speedup_end_to_end:.1f}x")


if __name__ == "__main__":
    main()
