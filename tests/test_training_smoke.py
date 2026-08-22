"""Test dymny (szybki, mala skala): trening MUSI zmniejszac strate wzgledem
losowej inicjalizacji - to nie sprawdza konkretnych liczb (patrz
khipu_neural/compare.py dla pelnych, zmierzonych wynikow), tylko ze
petla treningowa faktycznie uczy, a nie stoi w miejscu/rozjezdza sie."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from khipu_neural.data import ResonanceDataset
from khipu_neural.models import BaselinePairwiseMLP, KHIPUResonanceNet
from khipu_neural.train import train_baseline, train_khipu


def test_baseline_training_reduces_loss():
    rng = np.random.default_rng(0)
    ds = ResonanceDataset(d_embed=4, seq_len=6, seed=0)
    model = BaselinePairwiseMLP(d_embed=4, hidden=8, rng=rng)
    hist = train_baseline(model, ds, steps=60, batch_size=16, lr=0.05)
    assert hist[-1] < hist[0] * 0.5, f"strata nie spadla wystarczajaco: {hist[0]} -> {hist[-1]}"


def test_khipu_training_runs_and_produces_finite_loss():
    """KHIPU-model NIE musi tu bic baseline (patrz compare.py - w pelnym
    eksperymencie przegrywa) - ten test tylko pilnuje, ze petla treningowa
    dziala (nie rzuca, nie produkuje NaN/inf) po zmianach w kodzie."""
    rng = np.random.default_rng(0)
    ds = ResonanceDataset(d_embed=4, seq_len=6, seed=0)
    model = KHIPUResonanceNet(d_embed=4, rng=rng, temperature=3.0)
    hist = train_khipu(model, ds, steps=60, batch_size=16, lr=0.02)
    assert all(np.isfinite(h) for h in hist)
    assert len(hist) == 60
