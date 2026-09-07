"""
ablation.py — sprawdza SPRAWNOSC wyniku z compare.py (KHIPUResonanceNetMLP
bije baseline): czy przewaga bierze sie z (a) wiekszej liczby parametrow,
czy (b) samej dyskretyzacji State9, czy (c) czegos innego? Uruchom:
`python3 -m khipu_neural.ablation`

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, 4 ziarna losowosci [1,2,3,4], ta sama maszyna):
===========================================================================
Ustawienia: d_embed=8, seq_len=10, 400 krokow, batch=32, lr=0.02.

| Model                                          | parametry | test MAE (srednia +/- std)   |
|-------------------------------------------------|-----------|-------------------------------|
| BaselinePairwiseMLP (hidden=16)                 | 289       | 0.286 +/- 0.062               |
| BaselinePairwiseMLP DOPASOWANY (hidden=22)      | 397       | 0.260 +/- 0.035               |
| KHIPUResonanceNetMLP (State9 + MLP)             | 402       | 0.133 +/- 0.033               |
| KHIPUResonanceNetMLPNoQuant (bottleneck BEZ     | 402       | 0.142 +/- 0.049               |
|   kwantyzacji - ciagle t zamiast +/-1)          |           |                                |

Pojedyncze przebiegi (seed 1,2,3,4):
  baseline_289:   0.218, 0.247, 0.295, 0.383
  baseline_397:   0.235, 0.235, 0.248, 0.320
  khipu_mlp:      0.090, 0.142, 0.121, 0.180
  khipu_noquant:  0.104, 0.104, 0.136, 0.223

===========================================================================
UCZCIWY WNIOSEK (WAZNA KOREKTA wzgledem tego, co sugerowalo README przed
tym eksperymentem)
===========================================================================
1. WIEKSZA LICZBA PARAMETROW SAMA W SOBIE NIE TLUMACZY WYNIKU: baseline
   dopasowany do 397 parametrow (blisko 402 KHIPU) poprawia sie tylko
   nieznacznie (0.260 vs 0.286) - dalej ogromna przepasc do ~0.13-0.14.
   To WYKLUCZA hipoteze "KHIPU wygrywa bo ma wiecej wag".

2. DYSKRETYZACJA (twarda kwantyzacja +/-1 + warunek rownowagi F4-RED)
   SAMA W SOBIE NIE JEST ZRODLEM PRZEWAGI: wersja BEZ kwantyzacji
   (ciagly bottleneck 9-wymiarowy) wypada PRAKTYCZNIE TAK SAMO jak
   wersja Z kwantyzacja (0.142 vs 0.133 - w granicach bledu na 4
   ziarnach, khipu_mlp wygrywa na 3/4 ziaren, noquant na 1/4 - roznica
   NIE jest wyraznie rozstrzygnieta przy tej liczbie prob).

3. PRAWDZIWYM ZRODLEM PRZEWAGI JEST WASKIE GARDLO WYMIAROWE (bottleneck
   do 9 osi) SAMO W SOBIE, niezaleznie od tego czy jest dyskretyzowane
   czy nie. To dobrze znany efekt w uczeniu reprezentacji (information
   bottleneck, autoenkodery) - nie jest to cos unikalnego dla
   "geometrycznego"/State9 podejscia z KHIPU. Twarda kwantyzacja State9
   NIE SZKODZI (roznica z wersja ciagla jest w granicach szumu), ale
   tez wyraznie NIE POMAGA ponad to, co daje juz sama redukcja
   wymiarowosci.

Poprawiona interpretacja calego eksperymentu: KHIPUResonanceNetMLP bije
baseline nie dlatego, ze "dyskretna geometria State9" jest magiczna, tylko
dlatego, ze wymusza mniejszy, bardziej skoncentrowany bottleneck przed
porownaniem sasiadow - to samo osiagnalby dowolny inny bottleneck do 9
wymiarow, kwantyzowany lub nie. Dyskretyzacja State9 w tym konkretnym
zadaniu jest NEUTRALNA, nie kluczowa.
"""
from __future__ import annotations
import numpy as np
from .data import ResonanceDataset
from .models import BaselinePairwiseMLP, KHIPUResonanceNetMLP, KHIPUResonanceNetMLPNoQuant
from .train import train_baseline, train_khipu, evaluate


def count_params(params: dict) -> int:
    return sum(np.size(v) for v in params.values())


def run(seeds=(1, 2, 3, 4), steps: int = 400, batch_size: int = 32, lr: float = 0.02,
        d_embed: int = 8, seq_len: int = 10, verbose: bool = True):
    results = {"baseline_289": [], "baseline_397": [], "khipu_mlp": [], "khipu_noquant": []}

    for seed in seeds:
        ds_test = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=200 + seed)
        ds_ref = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed)
        ds_test.category_embed = ds_ref.category_embed.copy()

        rng = np.random.default_rng(seed)
        m = BaselinePairwiseMLP(d_embed=d_embed, hidden=16, rng=rng)
        train_baseline(m, ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed),
                        steps=steps, batch_size=batch_size, lr=lr)
        results["baseline_289"].append(evaluate(m, ds_test, n_samples=500))

        rng = np.random.default_rng(seed)
        m = BaselinePairwiseMLP(d_embed=d_embed, hidden=22, rng=rng)  # ~397 parametrow, dopasowane do KHIPU (402)
        train_baseline(m, ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed),
                        steps=steps, batch_size=batch_size, lr=lr)
        results["baseline_397"].append(evaluate(m, ds_test, n_samples=500))

        rng = np.random.default_rng(seed)
        m = KHIPUResonanceNetMLP(d_embed=d_embed, hidden=16, rng=rng)
        train_khipu(m, ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed),
                     steps=steps, batch_size=batch_size, lr=lr)
        results["khipu_mlp"].append(evaluate(m, ds_test, n_samples=500))

        rng = np.random.default_rng(seed)
        m = KHIPUResonanceNetMLPNoQuant(d_embed=d_embed, hidden=16, rng=rng)
        train_khipu(m, ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed),
                     steps=steps, batch_size=batch_size, lr=lr)
        results["khipu_noquant"].append(evaluate(m, ds_test, n_samples=500))

        if verbose:
            print(f"seed={seed}: base289={results['baseline_289'][-1]:.4f} "
                  f"base397={results['baseline_397'][-1]:.4f} "
                  f"khipu_mlp={results['khipu_mlp'][-1]:.4f} "
                  f"khipu_noquant={results['khipu_noquant'][-1]:.4f}")

    if verbose:
        print()
        for k, v in results.items():
            print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    return results


if __name__ == "__main__":
    run()
