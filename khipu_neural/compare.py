"""
compare.py — pelny eksperyment: BaselinePairwiseMLP vs KHIPUResonanceNet
na zadaniu "liczenie rezonansow" (data.py). Uruchom: `python3 -m khipu_neural.compare`

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, ta sama maszyna co reszta benchmarkow w KHIPU):
===========================================================================
Ustawienia: d_embed=8, seq_len=10, 4 kategorie ukryte, szum std=0.4, batch=32.

  Baseline (generyczny MLP par sasiadow, 289 parametrow), lr=0.02, 400 krokow:
      MSE: 2.66 -> 0.13 (koncowe), plynna, stabilna zbieznosc
      test MAE (500 probek): 0.365
      czas treningu: 5.4s

  KHIPUResonanceNet (kwantyzacja State9 + bramka rezonansu, 83 parametry),
  lr=0.02, 400 krokow:
      MSE: 33.2 -> 1.52 (koncowe), NIESTABILNA zbieznosc (np. skok w gore
      w polowie treningu)
      test MAE (500 probek): 1.08
      czas treningu: 26.0s (~5x wolniej niz baseline - koszt STE + petli
      po tokenach w czystym Pythonie)

  Trywialny model ("zawsze przewiduj srednia z treningu"): test MAE = 1.03

  Proba strojenia KHIPU (nizszy lr=0.005, temperatura sigmoid 2.0 zamiast
  4.0, 800 krokow zamiast 400): test MAE = 1.15 - NIE POPRAWILO wyniku,
  nadal w okolicach poziomu trywialnego predyktora.

UCZCIWY WNIOSEK: na tym konkretnym zadaniu generyczny MLP bez zadnej
"geometrycznej" struktury uczy sie zdecydowanie lepiej i szybciej niz
architektura inspirowana KHIPU. KHIPUResonanceNet w praktyce NIE PRZEBIJA
trywialnego predyktora sredniej, mimo ze zadanie zostalo zaprojektowane
WPROST pod regule GIPU (wykrywanie zgodnosci kategorii sasiadow). To NIE
jest dowod, ze taka architektura nigdy nie moze zadzialac - ale w tej
konkretnej, uczciwie przetestowanej formie, na tym zadaniu, PRZEGRYWA.
Prawdopodobne przyczyny (patrz README.md, sekcja "Dlaczego KHIPU-neural
przegrywa"): (1) ostateczna agregacja rezonansu to tylko 2 skalary
(w_resonant/w_other) - dużo mniejsza pojemnosc funkcyjna niz pelny MLP,
(2) twarda kwantyzacja + STE jest trudniejsza do optymalizacji niz
gladka reprezentacja ciagla, (3) mozliwe ze potrzeba dluzszego treningu/
lepszego strojenia niz sprawdzono tutaj.
"""
from __future__ import annotations
import time
import numpy as np
from .data import ResonanceDataset
from .models import BaselinePairwiseMLP, KHIPUResonanceNet
from .train import train_baseline, train_khipu, evaluate


def count_params(params: dict) -> int:
    return sum(np.size(v) for v in params.values())


def run(steps: int = 400, batch_size: int = 32, lr: float = 0.02,
        d_embed: int = 8, seq_len: int = 10, seed: int = 42, verbose: bool = True):
    ds_train = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=1)
    ds_test = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=2)
    ds_test.category_embed = ds_train.category_embed.copy()  # ta sama "prawda" dla obu zbiorow

    rng_base = np.random.default_rng(seed)
    rng_khipu = np.random.default_rng(seed)
    baseline = BaselinePairwiseMLP(d_embed=d_embed, hidden=16, rng=rng_base)
    khipu = KHIPUResonanceNet(d_embed=d_embed, rng=rng_khipu, temperature=4.0)

    if verbose:
        print(f"Parametry: baseline={count_params(baseline.params())}, "
              f"khipu={count_params(khipu.params())}")

    t0 = time.perf_counter()
    hist_base = train_baseline(baseline, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    hist_khipu = train_khipu(khipu, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    t_khipu = time.perf_counter() - t0

    mae_base = evaluate(baseline, ds_test, n_samples=500)
    mae_khipu = evaluate(khipu, ds_test, n_samples=500)

    X_tr, y_tr = ds_train.sample_batch(2000)
    mean_pred = float(np.mean(y_tr))
    X_te, y_te = ds_test.sample_batch(500)
    mae_trivial = float(np.mean(np.abs(mean_pred - y_te)))

    result = {
        "hist_base": hist_base, "hist_khipu": hist_khipu,
        "mae_base": mae_base, "mae_khipu": mae_khipu, "mae_trivial": mae_trivial,
        "t_base": t_base, "t_khipu": t_khipu,
    }

    if verbose:
        print(f"czas treningu: baseline={t_base:.1f}s, khipu={t_khipu:.1f}s")
        print(f"MSE koncowe: baseline={hist_base[-1]:.3f}, khipu={hist_khipu[-1]:.3f}")
        print(f"test MAE: baseline={mae_base:.4f}, khipu={mae_khipu:.4f}, "
              f"trywialny(srednia)={mae_trivial:.4f}")

    return result


if __name__ == "__main__":
    run()
