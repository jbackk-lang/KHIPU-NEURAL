"""
train.py — trenuje i porownuje BaselinePairwiseMLP oraz KHIPUResonanceNet
na zadaniu "liczenie rezonansow" (data.py). Reczny Adam (bez frameworka -
PyTorch nie dalo sie pobrac w tym sandboxie, siec byla zbyt wolna/zablokowana
dla download.pytorch.org). Wszystkie parametry modeli sa numpy-tablicami
(rowniez skalary, jako tablice 1-elementowe) - dzieki temu Adam moze je
aktualizowac PRAWDZIWA referencja (in-place), bez recznej synchronizacji.
"""
from __future__ import annotations
import numpy as np
from .data import ResonanceDataset
from .models import BaselinePairwiseMLP, KHIPUResonanceNet


class Adam:
    def __init__(self, params: dict, lr: float = 0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k, p in params.items():
            g = np.asarray(grads[k], dtype=np.float64)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # in-place, prawdziwa referencja


def _train(model, dataset: ResonanceDataset, steps: int, batch_size: int, lr: float):
    params = model.params()
    opt = Adam(params, lr=lr)
    history = []
    for step in range(steps):
        X, y = dataset.sample_batch(batch_size)
        total_loss = 0.0
        grad_accum = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        for b in range(batch_size):
            pred, cache = model.forward(X[b])
            err = pred - y[b]
            total_loss += err ** 2
            grads = model.backward(2 * err, cache)
            for k in grad_accum:
                grad_accum[k] += np.asarray(grads[k], dtype=np.float64) / batch_size
        opt.step(params, grad_accum)
        history.append(total_loss / batch_size)
    return history


def train_baseline(model: BaselinePairwiseMLP, dataset: ResonanceDataset,
                    steps: int, batch_size: int, lr: float):
    return _train(model, dataset, steps, batch_size, lr)


def train_khipu(model: KHIPUResonanceNet, dataset: ResonanceDataset,
                 steps: int, batch_size: int, lr: float):
    return _train(model, dataset, steps, batch_size, lr)


def evaluate(model, dataset: ResonanceDataset, n_samples: int):
    X, y = dataset.sample_batch(n_samples)
    errs = []
    for b in range(n_samples):
        pred, _ = model.forward(X[b])
        errs.append(abs(pred - y[b]))
    return float(np.mean(errs))
