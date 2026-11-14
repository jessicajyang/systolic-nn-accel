"""Train a small MLP on the scikit-learn digits dataset (8x8 images) and quantize it to int8.

The same integer forward pass runs either on plain NumPy (reference) or on the
accelerator model, so the two can be compared bit-for-bit.

Network:  64 -> 32 (ReLU) -> 10, int8 weights/activations, int32 accumulation.
Requantization between layers is done on the host with a float scale (a fixed-
point multiplier would replace this in a fully on-chip pipeline).
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def _sym_quant(w):
    scale = 127.0 / np.abs(w).max()
    return np.clip(np.round(w * scale), -127, 127).astype(np.int32), scale


def build(seed=0):
    x, y = load_digits(return_X_y=True)
    x = x / 16.0                                  # pixels 0..16 -> 0..1
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        max_iter=600, random_state=seed).fit(xtr, ytr)

    w1, w2 = clf.coefs_
    b1, b2 = clf.intercepts_
    s_x = 127.0                                   # input scale (0..1 -> 0..127)
    w1q, s_w1 = _sym_quant(w1)
    w2q, s_w2 = _sym_quant(w2)

    # calibrate the hidden-activation scale on the training set
    h_float = np.maximum(xtr @ w1 + b1, 0.0)
    s_h = 127.0 / h_float.max()

    return {
        "xte": xte, "yte": yte, "float_acc": clf.score(xte, yte),
        "s_x": s_x, "s_h": s_h,
        "w1q": w1q, "b1q": np.round(b1 * s_x * s_w1).astype(np.int32), "s_w1": s_w1,
        "w2q": w2q, "b2q": np.round(b2 * s_h * s_w2).astype(np.int32), "s_w2": s_w2,
    }


def quantize_inputs(m, x01):
    return np.clip(np.round(x01 * m["s_x"]), 0, 127).astype(np.int32)


def forward_int8(m, xq, matmul):
    """Integer forward pass. `matmul(a, b)` does the int8 x int8 -> int32 products."""
    acc1 = matmul(xq, m["w1q"]) + m["b1q"]
    acc1 = np.maximum(acc1, 0)                                           # ReLU
    scale = m["s_h"] / (m["s_x"] * m["s_w1"])
    hq = np.clip(np.round(acc1 * scale), 0, 127).astype(np.int32)        # requantize to int8
    return matmul(hq, m["w2q"]) + m["b2q"]                               # logits (int32)
