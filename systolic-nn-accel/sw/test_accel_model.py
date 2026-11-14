"""Self-checking tests for the cycle-accurate model (run: python sw/test_accel_model.py)."""
import numpy as np

from accel_model import Accelerator, SystolicArrayModel

rng = np.random.default_rng(1234)


def rand8(*shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int32)


def test_single_tile_exact_and_cycles():
    for n in (1, 2, 4, 8, 16):
        for k in (1, 2, n, n + 3, 3 * n):
            a, b = rand8(n, k), rand8(k, n)
            c, cyc = SystolicArrayModel(n).run_tile(a, b)
            assert np.array_equal(c, a @ b), (n, k)
            assert cyc == k + 2 * n - 2, (n, k, cyc)


def test_3n_minus_2_when_k_equals_n():
    for n in (2, 4, 8, 16):
        _, cyc = SystolicArrayModel(n).run_tile(rand8(n, n), rand8(n, n))
        assert cyc == 3 * n - 2


def test_int8_extremes_no_overflow():
    n, k = 8, 64
    for v in (-128, 127):
        a = np.full((n, k), v, dtype=np.int32)
        c, _ = SystolicArrayModel(n).run_tile(a, a.T.copy())
        assert np.array_equal(c, a @ a.T)


def test_tiled_matmul_arbitrary_sizes():
    for n in (4, 8):
        acc = Accelerator(n=n, kmax=16)
        for (m, k, p) in [(1, 1, 1), (5, 7, 3), (13, 40, 11), (17, 16, 9), (30, 100, 21)]:
            a, b = rand8(m, k), rand8(k, p)
            assert np.array_equal(acc.matmul(a, b), a @ b), (n, m, k, p)


def test_reuse_without_stale_state():
    arr = SystolicArrayModel(4)
    a1, b1 = rand8(4, 9), rand8(9, 4)
    a2, b2 = rand8(4, 5), rand8(5, 4)
    arr.run_tile(a1, b1)
    c2, _ = arr.run_tile(a2, b2)
    assert np.array_equal(c2, a2 @ b2)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)} model tests passed")
