"""Smoke + gradient checks for bottleneck_sweep.BottleneckMLP (parametryzowany
odpowiednik KHIPUResonanceNetMLP z konfigurowalna liczba osi bottlenecku).
Nie odtwarza calego sweepu (za wolne na test) - tylko potwierdza, ze
pipeline dziala i gradienty sa poprawne dla przykladowego n_axes != 9.
"""
import math
import numpy as np
from khipu_neural.bottleneck_sweep import BottleneckMLP, run_one
from khipu_neural.quantize import balance_correct


def _quantize(Wq, bq, x):
    proj = Wq @ x + bq
    t = np.tanh(proj)
    q = np.sign(t)
    q[q == 0] = 1.0
    q = balance_correct(q, t)
    return t, q


def test_run_one_produces_finite_mae():
    mae, n_params = run_one(n_axes=6, seed=1, steps=15)
    assert math.isfinite(mae)
    assert mae >= 0.0
    assert n_params > 0


def test_bottleneck_mlp_gradients_match_ste_numerical():
    rng = np.random.default_rng(99)
    d, T, hidden, n_axes = 3, 5, 4, 6
    model = BottleneckMLP(d_embed=d, n_axes=n_axes, hidden=hidden, rng=rng)
    seq = rng.normal(size=(T, d))

    pred, cache = model.forward(seq)
    grads = model.backward(1.0, cache)
    eps = 1e-5

    for name in ("W1", "b1", "w2", "b2"):
        param = getattr(model, name)
        flat = param.flatten().copy()
        num_grad = np.zeros_like(flat)
        for i in range(len(flat)):
            plus = flat.copy(); plus[i] += eps
            minus = flat.copy(); minus[i] -= eps

            def set_and_forward(v):
                old = getattr(model, name).copy()
                getattr(model, name)[...] = v.reshape(old.shape)
                p, _ = model.forward(seq)
                getattr(model, name)[...] = old
                return p

            num_grad[i] = (set_and_forward(plus) - set_and_forward(minus)) / (2 * eps)
        ana = grads[name].flatten()
        assert np.max(np.abs(num_grad - ana)) < 1e-4, name

    Wq0 = model.Wq.copy()
    bq0 = model.bq.copy()
    t0_list, q0_list = [], []
    for i in range(T):
        t, q = _quantize(Wq0, bq0, seq[i])
        t0_list.append(t); q0_list.append(q)

    def pred_from_codes(codes):
        p = 0.0
        for i in range(T - 1):
            z = np.concatenate([codes[i], codes[i + 1]])
            h = np.tanh(model.W1 @ z + model.b1)
            p += float(model.w2 @ h + model.b2[0])
        return p

    def numerical_grad_wq(idx):
        Wp = Wq0.copy(); Wp[idx] += eps
        Wm = Wq0.copy(); Wm[idx] -= eps
        codes_p, codes_m = [], []
        for i in range(T):
            tp, _ = _quantize(Wp, bq0, seq[i])
            tm, _ = _quantize(Wm, bq0, seq[i])
            codes_p.append(q0_list[i] + (tp - t0_list[i]))
            codes_m.append(q0_list[i] + (tm - t0_list[i]))
        return (pred_from_codes(codes_p) - pred_from_codes(codes_m)) / (2 * eps)

    num_grad_Wq = np.zeros_like(Wq0)
    it = np.nditer(Wq0, flags=["multi_index"])
    for _ in it:
        num_grad_Wq[it.multi_index] = float(numerical_grad_wq(it.multi_index))

    assert np.max(np.abs(num_grad_Wq - grads["Wq"])) < 1e-4
