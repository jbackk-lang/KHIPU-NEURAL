"""Testy dymne dla frozen_projection.py i dotproduct_task.py (testy 1 i 2
z ostatniej rundy weryfikacji sprawnosci bottlenecku - patrz docstringi
tych modulow dla pelnych, zmierzonych wynikow)."""
import math
import numpy as np
from khipu_neural.frozen_projection import FrozenBottleneckMLP
from khipu_neural.bottleneck_sweep import BottleneckMLP
from khipu_neural.dotproduct_task import DotProductDataset, DistanceDataset


def test_frozen_bottleneck_excludes_wq_bq_from_params():
    rng = np.random.default_rng(1)
    m = FrozenBottleneckMLP(d_embed=8, n_axes=9, hidden=16, rng=rng)
    assert set(m.params().keys()) == {"W1", "b1", "w2", "b2"}


def test_frozen_wq_bq_never_change_after_training_step():
    from khipu_neural.data import ResonanceDataset
    from khipu_neural.train import Adam

    rng = np.random.default_rng(2)
    m = FrozenBottleneckMLP(d_embed=8, n_axes=9, hidden=16, rng=rng)
    Wq_before, bq_before = m.Wq.copy(), m.bq.copy()

    ds = ResonanceDataset(d_embed=8, seq_len=10, seed=1)
    params = m.params()
    opt = Adam(params, lr=0.02)
    X, y = ds.sample_batch(8)
    grad_accum = {k: np.zeros_like(v) for k, v in params.items()}
    for b in range(8):
        pred, cache = m.forward(X[b])
        err = pred - y[b]
        grads = m.backward(2 * err, cache)
        for k in grad_accum:
            grad_accum[k] += grads[k] / 8
    opt.step(params, grad_accum)

    assert np.array_equal(m.Wq, Wq_before)
    assert np.array_equal(m.bq, bq_before)


def test_dotproduct_dataset_shapes_and_finiteness():
    ds = DotProductDataset(d_embed=8, seq_len=10, seed=0)
    X, y = ds.sample_batch(16)
    assert X.shape == (16, 10, 8)
    assert y.shape == (16,)
    assert np.all(np.isfinite(y))


def test_distance_dataset_shapes_and_nonnegative_labels():
    ds = DistanceDataset(d_embed=8, n_categories=4, seq_len=10, seed=0)
    X, y = ds.sample_batch(16)
    assert X.shape == (16, 10, 8)
    assert y.shape == (16,)
    assert np.all(y >= 0.0)   # suma kwadratow odleglosci - zawsze nieujemna
    assert np.all(np.isfinite(y))
