"""Cycle-accurate Python model of the systolic-array accelerator, plus a tiling driver.

`SystolicArrayModel` mirrors rtl/systolic_array.v register-for-register, so its
cycle counts and results are what the RTL produces. `Accelerator` is the
software driver: it tiles arbitrary int8 matrix multiplies onto the N x N array
and keeps performance statistics.
"""
from dataclasses import dataclass, field

import numpy as np


class SystolicArrayModel:
    """N x N output-stationary array. One call to `step` == one clock edge."""

    def __init__(self, n: int):
        self.n = n
        self.reset()

    def reset(self):
        n = self.n
        self.a_reg = np.zeros((n, n), dtype=np.int32)   # PE a_out registers
        self.b_reg = np.zeros((n, n), dtype=np.int32)   # PE b_out registers
        self.acc = np.zeros((n, n), dtype=np.int32)     # PE accumulators

    def step(self, a_edge, b_edge):
        """Advance one clock. a_edge[i] enters row i, b_edge[j] enters column j."""
        n = self.n
        a_in = np.empty((n, n), dtype=np.int32)
        b_in = np.empty((n, n), dtype=np.int32)
        a_in[:, 0] = a_edge
        a_in[:, 1:] = self.a_reg[:, :-1]
        b_in[0, :] = b_edge
        b_in[1:, :] = self.b_reg[:-1, :]
        # All PEs update simultaneously from the *old* register values.
        self.acc = (self.acc + a_in * b_in).astype(np.int32)   # int32 wraparound
        self.a_reg = a_in
        self.b_reg = b_in

    def run_tile(self, a_tile, b_tile):
        """C = A (n x K) @ B (K x n). Returns (C int32, cycles spent in the array)."""
        n = self.n
        k = a_tile.shape[1]
        assert a_tile.shape == (n, k) and b_tile.shape == (k, n)
        self.reset()                                      # the controller's CLEAR state
        cycles = k + 2 * n - 2
        for t in range(cycles):
            a_edge = np.zeros(n, dtype=np.int32)
            b_edge = np.zeros(n, dtype=np.int32)
            for i in range(n):                            # skew: lane i delayed i cycles
                kk = t - i
                if 0 <= kk < k:
                    a_edge[i] = a_tile[i, kk]
                    b_edge[i] = b_tile[kk, i]
            self.step(a_edge, b_edge)
        return self.acc.copy(), cycles


@dataclass
class Stats:
    compute_cycles: int = 0     # cycles the array is running
    transfer_cycles: int = 0    # register-bus cycles (1 word / cycle, no overlap)
    useful_macs: int = 0        # real multiply-accumulates (padding excluded)
    tiles: int = 0
    n: int = 0

    @property
    def baseline_cycles(self):
        """Idealised scalar core: 1 MAC per cycle, zero loop/load overhead."""
        return self.useful_macs

    @property
    def utilization(self):
        return self.useful_macs / (self.n * self.n * self.compute_cycles)

    @property
    def speedup_compute_only(self):
        return self.baseline_cycles / self.compute_cycles

    @property
    def speedup_end_to_end(self):
        return self.baseline_cycles / (self.compute_cycles + self.transfer_cycles)


@dataclass
class Accelerator:
    """Software driver: tiles A (M x K) @ B (K x P) onto the N x N array."""
    n: int = 8
    kmax: int = 64
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        self.array = SystolicArrayModel(self.n)
        self.stats.n = self.n

    def matmul(self, a, b):
        a = np.asarray(a, dtype=np.int32)
        b = np.asarray(b, dtype=np.int32)
        assert a.min() >= -128 and a.max() <= 127 and b.min() >= -128 and b.max() <= 127, \
            "operands must fit in int8"
        m, k = a.shape
        k2, p = b.shape
        assert k == k2
        n = self.n
        c = np.zeros((m, p), dtype=np.int32)
        for i0 in range(0, m, n):
            for j0 in range(0, p, n):
                for k0 in range(0, k, self.kmax):              # K chunks > KMAX
                    kc = min(self.kmax, k - k0)
                    a_t = np.zeros((n, kc), dtype=np.int32)    # zero-pad edge tiles
                    b_t = np.zeros((kc, n), dtype=np.int32)
                    ar = a[i0:i0 + n, k0:k0 + kc]
                    bc = b[k0:k0 + kc, j0:j0 + n]
                    a_t[:ar.shape[0], :] = ar
                    b_t[:, :bc.shape[1]] = bc
                    c_t, cyc = self.array.run_tile(a_t, b_t)
                    c[i0:i0 + ar.shape[0], j0:j0 + bc.shape[1]] += c_t[:ar.shape[0], :bc.shape[1]]
                    s = self.stats
                    s.compute_cycles += cyc
                    s.transfer_cycles += 2 * n * kc + n * n + 3   # A + B writes, C reads, ctrl
                    s.useful_macs += ar.shape[0] * bc.shape[1] * kc
                    s.tiles += 1
        return c
