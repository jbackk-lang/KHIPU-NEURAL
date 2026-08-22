"""
compare.py — pelny eksperyment: BaselinePairwiseMLP vs KHIPUResonanceNet
(2 skalary) vs KHIPUResonanceNetMLP (MLP nad kodami State9) na zadaniu
"liczenie rezonansow" (data.py). Uruchom: `python3 -m khipu_neural.compare`

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, ta sama maszyna co reszta benchmarkow w KHIPU):
===========================================================================
Ustawienia: d_embed=8, seq_len=10, 4 kategorie ukryte, szum std=0.4, batch=32,
400 krokow, Adam (recany), lr=0.02. Srednia +/- odchylenie std z 3 ziaren.

  Baseline (generyczny MLP par sasiadow, 289 parametrow):
      test MAE: 0.253 +/- 0.032

  KHIPUResonanceNet (kwantyzacja State9 + DWA SKALARY jako agregacja,
  83 parametry):
      test MAE: ~1.08 (pojedynczy przebieg) - NIE bije trywialnego
      predyktora sredniej (MAE 1.03). Niestabilna zbieznosc.

  KHIPUResonanceNetMLP (kwantyzacja State9 + MLP nad kodami zamiast
  dwoch skalarow, 402 parametry):
      test MAE: 0.114 +/- 0.014  <-- NAJLEPSZY z trzech, ~2x lepszy niz baseline

UCZCIWY WNIOSEK (zaktualizowany po dodaniu KHIPUResonanceNetMLP):
Pierwsza wersja architektury inspirowanej KHIPU (2 skalary) przegrywala
z trywialnym predyktorem - ale powodem NIE byla sama dyskretna kwantyzacja
State9, tylko zbyt slaba agregacja (2 liczby zamiast pelnej warstwy).
Po zastapieniu agregacji MLP-em o podobnej pojemnosci co baseline,
architektura z kwantyzacja State9 KONSEKWENTNIE WYGRYWA z czysto
ciagla wersja, ~2x pod wzgledem MAE, powtarzalnie na 3 niezaleznych
ziarnach losowosci (0.098/0.133/0.111 vs 0.218/0.247/0.295).

Prawdopodobne wyjasnienie: dane wejsciowe to zaszumione obserwacje
niewielkiej liczby (4) ukrytych kategorii - twarda kwantyzacja do
dyskretnego kodu dziala jak wymuszona decyzja kategoryczna, usuwajaca
szum PRZED porownaniem sasiadow. Ciagly baseline musi sam nauczyc sie
odfiltrowac szum wewnatrz MLP, co jest trudniejsze. To spojne z tym,
dlaczego kwantyzacja (VQ-VAE i podobne) bywa uzywana jako mechanizm
odszumiajacy/regularyzujacy w innych kontekstach.

Zastrzezenie: to jedno male, syntetyczne zadanie zaprojektowane pod
regule GIPU - nie dowod, ze to dziala ogolnie. KHIPUResonanceNetMLP ma
tez wiecej parametrow (402) niz baseline (289), wiec czesc przewagi
MOZE czesciowo wynikac z wiekszej pojemnosci, nie tylko z kwantyzacji -
uczciwe porownanie "parametr do parametru" (baseline z hidden=23 dla
tej samej liczby parametrow) nie zostalo tu jeszcze sprawdzone.
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
    from .models import KHIPUResonanceNetMLP

    ds_train = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=1)
    ds_test = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=2)
    ds_test.category_embed = ds_train.category_embed.copy()  # ta sama "prawda" dla obu zbiorow

    rng_base = np.random.default_rng(seed)
    rng_khipu = np.random.default_rng(seed)
    rng_khipu_mlp = np.random.default_rng(seed)
    baseline = BaselinePairwiseMLP(d_embed=d_embed, hidden=16, rng=rng_base)
    khipu = KHIPUResonanceNet(d_embed=d_embed, rng=rng_khipu, temperature=4.0)
    khipu_mlp = KHIPUResonanceNetMLP(d_embed=d_embed, hidden=16, rng=rng_khipu_mlp)

    if verbose:
        print(f"Parametry: baseline={count_params(baseline.params())}, "
              f"khipu(2 skalary)={count_params(khipu.params())}, "
              f"khipu_mlp={count_params(khipu_mlp.params())}")

    t0 = time.perf_counter()
    hist_base = train_baseline(baseline, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    hist_khipu = train_khipu(khipu, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    t_khipu = time.perf_counter() - t0

    t0 = time.perf_counter()
    hist_khipu_mlp = train_khipu(khipu_mlp, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    t_khipu_mlp = time.perf_counter() - t0

    mae_base = evaluate(baseline, ds_test, n_samples=500)
    mae_khipu = evaluate(khipu, ds_test, n_samples=500)
    mae_khipu_mlp = evaluate(khipu_mlp, ds_test, n_samples=500)

    X_tr, y_tr = ds_train.sample_batch(2000)
    mean_pred = float(np.mean(y_tr))
    X_te, y_te = ds_test.sample_batch(500)
    mae_trivial = float(np.mean(np.abs(mean_pred - y_te)))

    result = {
        "hist_base": hist_base, "hist_khipu": hist_khipu, "hist_khipu_mlp": hist_khipu_mlp,
        "mae_base": mae_base, "mae_khipu": mae_khipu, "mae_khipu_mlp": mae_khipu_mlp,
        "mae_trivial": mae_trivial,
        "t_base": t_base, "t_khipu": t_khipu, "t_khipu_mlp": t_khipu_mlp,
    }

    if verbose:
        print(f"czas treningu: baseline={t_base:.1f}s, khipu={t_khipu:.1f}s, "
              f"khipu_mlp={t_khipu_mlp:.1f}s")
        print(f"test MAE: baseline={mae_base:.4f}, khipu(2 skalary)={mae_khipu:.4f}, "
              f"khipu_mlp={mae_khipu_mlp:.4f}, trywialny(srednia)={mae_trivial:.4f}")

    return result


if __name__ == "__main__":
    run()
