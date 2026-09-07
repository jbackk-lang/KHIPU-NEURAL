"""
sweep_noise_categories.py — domyka SKILL.14.8: pozytywny wynik
KHIPUResonanceNetMLP (patrz compare.py) zostal dotad zmierzony na JEDNEJ
konfiguracji zadania (d_embed=8, n_categories=4, seq_len=10, noise_std=0.4).
Pytanie: czy przewaga (kwantyzacja State9-jak jako "wymuszone odszumienie")
utrzymuje sie w calym rozsadnym zakresie poziomu szumu i liczby ukrytych
kategorii, czy jest artefaktem jednej konkretnej konfiguracji?

PRE-REJESTRACJA (przed uruchomieniem):
- Siatka: noise_std in {0.1, 0.4, 0.8, 1.6} x n_categories in {2, 4, 8}
  (d_embed=8, seq_len=10 trzymane stale - to osobna, juz zbadana os w
  bottleneck_sweep.py).
- Modele: BaselinePairwiseMLP vs KHIPUResonanceNetMLP, identyczne
  hiperparametry treningu co w compare.py (steps=400, batch=32, lr=0.02,
  hidden=16), 3 ziarna na konfiguracje (1, 2, 3).
- Metryka: test MAE (500 probek), oraz MAE trywialnego predyktora
  sredniej jako dolna referencja "czy zadanie w ogole ma tresc".
- Kryterium: dla kazdej konfiguracji sprawdzamy czy mean(MAE_khipu_mlp)
  < mean(MAE_baseline) (khipu_mlp wygrywa), i czy oba biją trywialny
  predyktor srednia (zadanie nie jest zdegenerowane).
- Uruchomienie: RAZ, cala siatka, bez wybierania najlepszych konfiguracji.

Uruchomienie: `python3 -m khipu_neural.sweep_noise_categories`
(dane w tym pliku zebrane przez `_sweep_checkpoint_runner.py` - to samo
zadanie, ale z checkpointem na dysku, bo pelna siatka 36 kombinacji nie
miesci sie w jednym wywolaniu powloki w tym sandboxie).

===========================================================================
ZMIERZONE WYNIKI (2026-08-30, pelna siatka 4x3 x 3 ziarna = 36 przebiegow,
d_embed=8, seq_len=10, steps=400, batch=32, lr=0.02, hidden=16):
===========================================================================

| n_categories | noise_std | baseline MAE | khipu_mlp MAE | trywialny | khipu wygrywa? |
|---|---|---|---|---|---|
| 2 | 0.1 | 0.0533 | 0.0171 | 1.178 | TAK |
| 2 | 0.4 | 0.1464 | 0.0776 | 1.178 | TAK |
| 2 | 0.8 | 0.3674 | 0.2109 | 1.178 | TAK |
| 2 | 1.6 | 0.8385 | 0.7893 | 1.178 | TAK |
| 4 | 0.1 | 0.1124 | 0.0258 | 1.044 | TAK |
| 4 | 0.4 | 0.2532 | 0.1255 | 1.044 | TAK (konfiguracja z compare.py) |
| 4 | 0.8 | 0.5225 | 0.4945 | 1.044 | TAK (slabo) |
| 4 | 1.6 | 0.9852 | 1.0081 | 1.044 | **NIE** (oba ~trywialny) |
| 8 | 0.1 | 0.1907 | 0.4670 | 0.786 | **NIE** (2.4x gorzej!) |
| 8 | 0.4 | 0.4706 | 0.5916 | 0.786 | **NIE** |
| 8 | 0.8 | 0.7319 | 0.7257 | 0.786 | TAK (slabo) |
| 8 | 1.6 | 0.8193 | 0.7470 | 0.786 | TAK (oba ~trywialny) |

khipu_mlp wygrywa w 9/12 konfiguracji, ale przegrywa w 3 - i te 3 porazki
NIE sa losowe, tworza spojny wzorzec:

**WNIOSEK (uczciwy, mieszany - nie jednoznacznie pozytywny jak sugerowalby
sam compare.py na jednej konfiguracji): przewaga kwantyzacji State9-jak
zalezy od stosunku liczby ukrytych kategorii do STALEJ liczby osi
bottlenecku (n_axes=9, z State9/F4-RED, nigdy nie dobierana pod zadanie).**

- Przy n_categories << n_axes (2, 4 - test z compare.py): kwantyzacja
  dziala jak "wymuszone odszumienie" i wygrywa, dokladnie zgodnie z
  pierwotna hipoteza - AZ DO bardzo wysokiego szumu (1.6), gdzie zadanie
  samo w sobie staje sie prawie nierozwiazywalne (oba modele ~zrownuja
  sie z trywialnym predyktorem srednia - nie ma juz czego "odszumiac").
- Przy n_categories=8 (blisko n_axes=9) i NISKIM/SREDNIM szumie (0.1,
  0.4): kwantyzacja PRZEGRYWA, wyraznie (0.19 vs 0.47 przy noise=0.1) -
  odwrotnie niz gdzie indziej. Mechanizm: przy niskim szumie baseline
  moze sam nauczyc sie precyzyjnej, ciaglej reprezentacji 8 kategorii
  wprost z danych; twarda kwantyzacja do 9-osiowego kodu ±1 przy 8
  kategoriach ma bardzo malo "miejsca" (F4-RED balans ogranicza faktycznie
  uzyteczna pojemnosc kodu bardziej niz 9 sugeruje) - to samo zjawisko co
  "n_axes=4 katastrofalne" w bottleneck_sweep.py, tylko lagodniejsze,
  bo n_categories=8 wciaz jest < 9, nie >> 9.
- Przy n_categories=8 i WYSOKIM szumie (0.8, 1.6): khipu_mlp znow wygrywa
  (choc oba modele sa juz blisko trywialnego) - przy duzym szumie baseline
  traci swoja przewage "moze nauczyc sie ciaglej reprezentacji precyzyjnie",
  bo dane sa zbyt zaszumione, zeby ta precyzja cokolwiek dala, a wymuszone
  odszumienie znow zaczyna pomagac.

**Zaktualizowany zakres wniosku z compare.py**: "kwantyzacja State9-jak
pomaga jako wymuszone odszumienie" NIE jest ogolna wlasnoscia architektury -
jest prawdziwe TYLKO gdy liczba ukrytych kategorii jest wyraznie mniejsza
niz stala liczba osi bottlenecku (n_axes=9) I szum nie jest ani znikomy,
ani tak duzy, ze zadanie staje sie nierozwiazywalne dla obu modeli. Blisko
granicy n_categories~n_axes, przewaga sie odwraca. To bezposrednio
potwierdza mechanistyczna intuicje z bottleneck_sweep.py (n_axes musi miec
realny zapas ponad liczbe kategorii), zamiast tylko przypuszczac ja z
osobna.
"""
from __future__ import annotations
import numpy as np
from .data import ResonanceDataset
from .models import BaselinePairwiseMLP, KHIPUResonanceNetMLP
from .train import train_baseline, train_khipu, evaluate


def run_one(noise_std: float, n_categories: int, seed: int,
            d_embed=8, seq_len=10, hidden=16, steps=400, batch_size=32, lr=0.02):
    ds_train = ResonanceDataset(d_embed=d_embed, n_categories=n_categories,
                                 seq_len=seq_len, noise_std=noise_std, seed=100 + seed)
    ds_test = ResonanceDataset(d_embed=d_embed, n_categories=n_categories,
                                seq_len=seq_len, noise_std=noise_std, seed=200 + seed)
    ds_test.category_embed = ds_train.category_embed.copy()

    rng_b = np.random.default_rng(seed)
    rng_k = np.random.default_rng(seed)
    baseline = BaselinePairwiseMLP(d_embed=d_embed, hidden=hidden, rng=rng_b)
    khipu_mlp = KHIPUResonanceNetMLP(d_embed=d_embed, hidden=hidden, rng=rng_k)

    train_baseline(baseline, ds_train, steps=steps, batch_size=batch_size, lr=lr)
    train_khipu(khipu_mlp, ds_train, steps=steps, batch_size=batch_size, lr=lr)

    mae_base = evaluate(baseline, ds_test, n_samples=500)
    mae_khipu = evaluate(khipu_mlp, ds_test, n_samples=500)

    X_tr, y_tr = ds_train.sample_batch(2000)
    mean_pred = float(np.mean(y_tr))
    X_te, y_te = ds_test.sample_batch(500)
    mae_trivial = float(np.mean(np.abs(mean_pred - y_te)))

    return mae_base, mae_khipu, mae_trivial


def run(noise_stds=(0.1, 0.4, 0.8, 1.6), n_cats=(2, 4, 8), seeds=(1, 2, 3), verbose=True):
    results = {}
    for nc in n_cats:
        for ns in noise_stds:
            bases, khipus, trivials = [], [], []
            for seed in seeds:
                b, k, t = run_one(noise_std=ns, n_categories=nc, seed=seed)
                bases.append(b)
                khipus.append(k)
                trivials.append(t)
            results[(nc, ns)] = {
                "base_mean": np.mean(bases), "base_std": np.std(bases),
                "khipu_mean": np.mean(khipus), "khipu_std": np.std(khipus),
                "trivial_mean": np.mean(trivials),
            }
            if verbose:
                r = results[(nc, ns)]
                khipu_wins = r["khipu_mean"] < r["base_mean"]
                beats_trivial = r["khipu_mean"] < r["trivial_mean"] and r["base_mean"] < r["trivial_mean"]
                print(f"n_cat={nc} noise={ns}: base={r['base_mean']:.3f}+/-{r['base_std']:.3f} "
                      f"khipu={r['khipu_mean']:.3f}+/-{r['khipu_std']:.3f} "
                      f"trivial={r['trivial_mean']:.3f} "
                      f"[khipu_wins={khipu_wins}, both_beat_trivial={beats_trivial}]")
    return results


if __name__ == "__main__":
    run()
