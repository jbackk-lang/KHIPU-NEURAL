"""Sprawdza, ze VectorizedBottleneckMLP daje IDENTYCZNE (do precyzji
zmiennoprzecinkowej) predykcje i gradienty co BottleneckMLP (petla
Python) - to nie jest gradient-check (matematyke juz sprawdzono w
tests/test_bottleneck_sweep_smoke.py), tylko test row0waznosci dwoch
implementacji tej samej matematyki."""
import numpy as np
from khipu_neural.bottleneck_sweep import BottleneckMLP
from khipu_neural.vectorized import VectorizedBottleneckMLP, benchmark


def test_vectorized_matches_loop_predictions_and_gradients():
    d_embed, n_axes, hidden, T, B = 8, 9, 16, 10, 6
    rng = np.random.default_rng(7)
    X = rng.normal(size=(B, T, d_embed))
    y = rng.normal(size=B)

    rng_m = np.random.default_rng(123)
    m_loop = BottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng_m)
    rng_m2 = np.random.default_rng(123)
    m_vec = VectorizedBottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng_m2)

    for k in m_loop.params():
        assert np.allclose(m_loop.params()[k], m_vec.params()[k])

    preds_loop = []
    grads_loop_accum = {k: np.zeros_like(v) for k, v in m_loop.params().items()}
    for b in range(B):
        pred, cache = m_loop.forward(X[b])
        preds_loop.append(pred)
        err = pred - y[b]
        grads = m_loop.backward(2 * err, cache)
        for k in grads_loop_accum:
            grads_loop_accum[k] += grads[k]
    preds_loop = np.array(preds_loop)

    pred_vec, cache_vec = m_vec.forward_batch(X)
    err_vec = pred_vec - y
    grads_vec = m_vec.backward_batch(2 * err_vec, cache_vec)

    assert np.max(np.abs(preds_loop - pred_vec)) < 1e-8
    for k in grads_loop_accum:
        assert np.max(np.abs(grads_loop_accum[k] - grads_vec[k])) < 1e-6, k


def test_vectorized_is_faster_than_loop():
    """Nie sprawdza konkretnej liczby (zalezy od maszyny) - tylko ze
    wektoryzacja jest szybsza, nie wolniejsza (regresja narzedziowa)."""
    t_loop, t_vec, speedup = benchmark(steps=10, batch_size=16)
    assert speedup > 1.0
