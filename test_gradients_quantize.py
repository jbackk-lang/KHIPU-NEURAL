import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from khipu_neural.quantize import State9Bottleneck, balance_correct, N_AXES


def test_balance_correct_always_satisfies_constraint():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.normal(size=N_AXES)
        q = np.sign(t)
        q[q == 0] = 1.0
        corrected = balance_correct(q, t)
        assert set(np.unique(corrected)) <= {-1.0, 1.0}
        assert abs(int(corrected.sum())) <= 1


def test_forward_output_always_valid_f4_red_state():
    rng = np.random.default_rng(1)
    layer = State9Bottleneck(d_in=6, rng=rng)
    for _ in range(100):
        x = rng.normal(size=6)
        q = layer.forward(x)
        assert q.shape == (N_AXES,)
        assert set(np.unique(q)) <= {-1.0, 1.0}
        assert abs(int(q.sum())) <= 1


def test_backward_matches_numerical_gradient_wq_bq():
    """Kluczowy test: reczny backward() MUSI zgadzac sie z numerycznym
    gradientem (STE oznacza, ze gradient plynie tak jakby q==t, wiec
    liczymy numeryczna pochodna funkcji f(Wq,bq) = sum(w * t(Wq,bq,x)),
    NIE przez twarda funkcje schodkowa sign+correct (ktora ma gradient
    zero prawie wszedzie) - to jest zamierzona, standardowa wlasciwosc STE,
    testujemy wiec pochodna wzgledem `t`, czyli wzgledem "pre-kwantyzacji"."""
    rng = np.random.default_rng(2)
    d = 5
    layer = State9Bottleneck(d_in=d, rng=rng)
    x = rng.normal(size=d)
    w = rng.normal(size=N_AXES)  # losowa "strata" liniowa po wyjsciu

    q = layer.forward(x)
    dL_dq = w.copy()  # dL/dq = w, bo L = sum(w*q)
    dL_dx, grads = layer.backward(dL_dq)

    eps = 1e-5

    def loss_with_wq(Wq_flat):
        old = layer.Wq.copy()
        layer.Wq = Wq_flat.reshape(layer.Wq.shape)
        t = np.tanh(layer.Wq @ x + layer.bq)  # STE: liczymy L wzgledem t, nie q
        L = float(np.dot(w, t))
        layer.Wq = old
        return L

    Wq_flat = layer.Wq.flatten().copy()
    num_grad = np.zeros_like(Wq_flat)
    for i in range(len(Wq_flat)):
        plus = Wq_flat.copy(); plus[i] += eps
        minus = Wq_flat.copy(); minus[i] -= eps
        num_grad[i] = (loss_with_wq(plus) - loss_with_wq(minus)) / (2 * eps)

    ana_grad = grads["Wq"].flatten()
    max_err = np.max(np.abs(num_grad - ana_grad))
    assert max_err < 1e-4, f"Gradient Wq nie zgadza sie z numerycznym, max_err={max_err}"


def test_backward_matches_numerical_gradient_input_x():
    rng = np.random.default_rng(3)
    d = 4
    layer = State9Bottleneck(d_in=d, rng=rng)
    x = rng.normal(size=d)
    w = rng.normal(size=N_AXES)

    q = layer.forward(x)
    dL_dx, grads = layer.backward(w.copy())

    eps = 1e-5

    def loss_with_x(xv):
        t = np.tanh(layer.Wq @ xv + layer.bq)
        return float(np.dot(w, t))

    num_grad = np.zeros_like(x)
    for i in range(len(x)):
        plus = x.copy(); plus[i] += eps
        minus = x.copy(); minus[i] -= eps
        num_grad[i] = (loss_with_x(plus) - loss_with_x(minus)) / (2 * eps)

    max_err = np.max(np.abs(num_grad - dL_dx))
    assert max_err < 1e-4, f"Gradient x nie zgadza sie z numerycznym, max_err={max_err}"
