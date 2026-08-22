import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from khipu_neural.models import BaselinePairwiseMLP, KHIPUResonanceNet, sigmoid
from khipu_neural.quantize import N_AXES


def test_baseline_gradients_match_numerical():
    rng = np.random.default_rng(10)
    d, T, hidden = 4, 6, 5
    model = BaselinePairwiseMLP(d_embed=d, hidden=hidden, rng=rng)
    seq = rng.normal(size=(T, d))

    pred, cache = model.forward(seq)
    grads = model.backward(1.0, cache)  # dL/dpred = 1 -> L = pred

    eps = 1e-5
    for name in ("W1", "b1", "w2", "b2"):
        param = getattr(model, name)
        flat = np.atleast_1d(param).flatten().copy()
        num_grad = np.zeros_like(flat)
        for i in range(len(flat)):
            plus = flat.copy(); plus[i] += eps
            minus = flat.copy(); minus[i] -= eps

            def set_and_forward(v):
                old = getattr(model, name)
                shp = np.shape(old)
                setattr(model, name, v.reshape(shp) if shp else float(v[0]))
                p, _ = model.forward(seq)
                setattr(model, name, old)
                return p

            num_grad[i] = (set_and_forward(plus) - set_and_forward(minus)) / (2 * eps)

        ana = np.atleast_1d(grads[name]).flatten()
        max_err = np.max(np.abs(num_grad - ana))
        assert max_err < 1e-4, f"{name}: max_err={max_err}"


def test_khipu_w_resonant_w_other_gradients_match_numerical():
    """w_resonant/w_other nie przechodza przez STE - kody sa funkcja
    WYLACZNIE Wq/bq (trzymanych na stale), wiec to jest zwykly, dokladny
    gradient bez zadnych przyblizen."""
    rng = np.random.default_rng(11)
    d, T = 4, 6
    model = KHIPUResonanceNet(d_embed=d, rng=rng)
    seq = rng.normal(size=(T, d))

    pred, cache = model.forward(seq)
    grads = model.backward(1.0, cache)

    eps = 1e-5
    for name in ("w_resonant", "w_other"):
        old = getattr(model, name)
        setattr(model, name, old + eps)
        p_plus, _ = model.forward(seq)
        setattr(model, name, old - eps)
        p_minus, _ = model.forward(seq)
        setattr(model, name, old)
        num_grad = (p_plus - p_minus) / (2 * eps)
        ana = float(grads[name].reshape(-1)[0]) if np.ndim(grads[name]) else float(grads[name])
        assert abs(num_grad - ana) < 1e-4, f"{name}: num={num_grad} ana={ana}"


def _quantize(Wq, bq, x):
    from khipu_neural.quantize import balance_correct
    proj = Wq @ x + bq
    t = np.tanh(proj)
    q = np.sign(t); q[q == 0] = 1.0
    q = balance_correct(q, t)
    return t, q


def test_khipu_wq_bq_gradients_match_ste_numerical():
    """Wq/bq przechodza przez STE na PRODUKCIE dwoch niezaleznie
    skwantyzowanych wektorow (agree = dot(q_i, q_i+1)). Poprawna definicja
    STE to `y = t + stop_gradient(q - t)`: FORWARD y == q (twarde), ale
    d(y)/d(t) == 1. Zeby to zweryfikowac numerycznie trzeba PRZY PERTURBACJI
    Wq policzyc y_perturbed = q_original + (t_perturbed - t_original) -
    czyli zlinearyzowac WOKOL oryginalnego punktu (q_original,t_original),
    NIE podstawiac czystego t wszedzie (to byla pierwsza, blednie napisana
    wersja tego testu - dawala niezerowy blad ~0.03-0.18, bo nie
    odzwierciedlala poprawnie stop_gradient - zobacz git log)."""
    rng = np.random.default_rng(12)
    d, T = 3, 5
    model = KHIPUResonanceNet(d_embed=d, rng=rng)
    seq = rng.normal(size=(T, d))

    Wq0 = model.bottleneck.Wq.copy()
    bq0 = model.bottleneck.bq.copy()

    t0_list, q0_list = [], []
    for i in range(T):
        t, q = _quantize(Wq0, bq0, seq[i])
        t0_list.append(t); q0_list.append(q)

    pred, cache = model.forward(seq)
    grads = model.backward(1.0, cache)

    def pred_from_codes(codes):
        p = 0.0
        for i in range(T - 1):
            a = float(np.dot(codes[i], codes[i + 1])) / N_AXES
            g = sigmoid(model.temperature * a)
            p += g * model.w_resonant + (1 - g) * model.w_other
        return p

    eps = 1e-5

    def numerical_grad_wq(idx):
        Wp = Wq0.copy(); Wp[idx] += eps
        Wm = Wq0.copy(); Wm[idx] -= eps
        codes_p, codes_m = [], []
        for i in range(T):
            tp, _ = _quantize(Wp, bq0, seq[i])
            tm, _ = _quantize(Wm, bq0, seq[i])
            codes_p.append(q0_list[i] + (tp - t0_list[i]))   # y = q_orig + (t_pert - t_orig)
            codes_m.append(q0_list[i] + (tm - t0_list[i]))
        return (pred_from_codes(codes_p) - pred_from_codes(codes_m)) / (2 * eps)

    num_grad_Wq = np.zeros_like(Wq0)
    it = np.nditer(Wq0, flags=["multi_index"])
    for _ in it:
        num_grad_Wq[it.multi_index] = float(numerical_grad_wq(it.multi_index))

    max_err_Wq = np.max(np.abs(num_grad_Wq - grads["Wq"]))
    assert max_err_Wq < 1e-4, f"Wq (STE) max_err={max_err_Wq}"
