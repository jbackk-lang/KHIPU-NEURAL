"""
bottleneck_sweep.py — testuje wprost wlasna hipoteze z ablation.py: skoro
przewaga KHIPUResonanceNetMLP nad baseline bierze sie GLOWNIE z redukcji
wymiaru (bottleneck), a nie z dyskretnosci State9, to CZY 9 osi (wziete z
State9/F4-RED w KHIPU, nie dobrane pod to zadanie) jest w ogole dobrym
wyborem? Zadanie ma tylko 4 ukryte kategorie - 9 to spory zapas.

Model tutaj to dokladnie architektura KHIPUResonanceNetMLP (kwantyzacja
State9-jak + MLP nad konkatenacja), ale z KONFIGUROWALNA liczba osi
bottlenecku zamiast na sztywno N_AXES=9 z quantize.py. Uzywa tej samej,
juz zweryfikowanej (w tests/test_gradients_models.py) matematyki STE -
tylko sparametryzowanej.

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, ziarna [1,2,3], d_embed=8, seq_len=10,
4 kategorie ukryte, 400 krokow, batch=32, lr=0.02, hidden=16):
===========================================================================

| osie bottlenecku | parytet | parametry | test MAE (srednia +/- std)   |
|---|---|---|---|
| 4  | parzysty  | 197 | 0.821 +/- 0.076  (KATASTROFA)          |
| 6  | parzysty  | 279 | 0.161 +/- 0.104  (duza wariancja)       |
| 9  | nieparzysty | 402 | 0.113 +/- 0.015  (referencja: KHIPUResonanceNetMLP) |
| 13 | nieparzysty | 566 | **0.100 +/- 0.008**  (najlepszy i najstabilniejszy) |
| 16 | parzysty  | 689 | 0.135 +/- 0.031  (gorzej niz 9 i 13)     |

WNIOSEK 1a (CZESC OBALONA PONIZEJ - patrz AKTUALIZACJA): na 3 ziarnach
wygladalo na to, ze 9 osi NIE jest optymalne i 13 wypada lepiej - ta
konkretna czesc wniosku NIE PRZETRWALA 5. ziarna (patrz nizej).

WNIOSEK 1b (PRZETRWAL AKTUALIZACJE): zbyt malo osi (4) jest
katastrofalne - za mala pojemnosc, zeby w ogole rozroznic 4 kategorie
pod kwantyzacja. Ta czesc wniosku jest solidna niezaleznie od tego, czy
9 czy 13 jest "lepsze" - obie sa wyraznie lepsze niz 4.

WNIOSEK 2 (POSTAWIONY NA 3 ZIARNACH, POTEM OBALONY NA 5 - patrz nizej):
na samych 3 ziarnach wygladalo na to, ze liczby NIEPARZYSTE (9, 13)
wypadaja lepiej i stabilniej niz sasiednie parzyste (6, 16), z mozliwym
wyjasnieniem mechanistycznym (warunek F4-RED |suma|<=1 jest luzniejszy
dla n nieparzystego). To byl dokladnie taki wniosek, przed ktorym
przestrzega caly ten projekt (patrz ablation.py) - i faktycznie okazal
sie PRZEDWCZESNY.

===========================================================================
AKTUALIZACJA (self_validate.py, 5 ziaren zamiast 3, 2026-08-22):
===========================================================================
Proces samodoskonalenia (khipu_neural/self_validate.py::self_improve_bottleneck)
dolozyl ziarna 4 i 5 dla n_axes=9 i n_axes=13 i wynik SIE ODWROCIL:

| n_axes | mean (3 ziarna) | mean (5 ziaren) |
|---|---|---|
| 9  | 0.113 | **0.134** (bez zmian rangi, ale wyzej) |
| 13 | 0.100 | **0.159** (bylo "najlepsze", teraz WYRAZNIE gorsze) |

Na 5 ziarnach n_axes=13 (std=0.076) jest GORSZE i DUZO mniej stabilne
niz n_axes=9 (std=0.029) - dokladne odwrocenie wniosku z 3 ziaren.
Mechanizm self_improve_bottleneck wymaga min. 5 ziaren wlasnie z tego
powodu i PRAWIDLOWO odrzucil promocje na n_axes=13 (decyzja: "keep:9").
Gdyby nie ten wymog, ten skrypt (albo jakikolwiek automat dzialajacy na
3 ziarnach) "poprawilby" model w zlym kierunku. WNIOSEK: 9 osi
(oryginalny wybor z State9/F4-RED) NIE zostal pobity przez zaden
przetestowany tu wariant - hipoteza o nieparzystosci NIE POTWIERDZONA
(byla artefaktem malej proby). Pelna historia decyzji:
self_improve_log.json (repo root).

Uruchomienie: `python3 -m khipu_neural.bottleneck_sweep`
(albo, z bramkami i logowaniem: `python3 -m khipu_neural.self_validate`)
"""
from __future__ import annotations
import numpy as np
from .quantize import balance_correct
from .data import ResonanceDataset
from .train import train_khipu, evaluate


class BottleneckMLP:
    """Jak KHIPUResonanceNetMLP, ale z n_axes jako parametrem zamiast
    sztywnego N_AXES=9 z quantize.py."""

    def __init__(self, d_embed: int, n_axes: int, hidden: int, rng: np.random.Generator):
        self.n_axes = n_axes
        scale_q = 1.0 / np.sqrt(d_embed)
        self.Wq = rng.normal(0, scale_q, size=(n_axes, d_embed))
        self.bq = np.zeros(n_axes)

        d_in = 2 * n_axes
        scale1 = 1.0 / np.sqrt(d_in)
        scale2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, scale1, size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, size=hidden)
        self.b2 = np.zeros(1)

    def params(self):
        return {"Wq": self.Wq, "bq": self.bq, "W1": self.W1, "b1": self.b1,
                "w2": self.w2, "b2": self.b2}

    def _quantize(self, x):
        proj = self.Wq @ x + self.bq
        t = np.tanh(proj)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct(q, t)
        return t, q

    def forward(self, seq: np.ndarray):
        T = seq.shape[0]
        codes, ts = [], []
        for i in range(T):
            t, q = self._quantize(seq[i])
            codes.append(q)
            ts.append(t)
        cache = {"codes": codes, "ts": ts, "seq": seq, "T": T, "z": [], "h": []}
        pred = 0.0
        for i in range(T - 1):
            z = np.concatenate([codes[i], codes[i + 1]])
            h = np.tanh(self.W1 @ z + self.b1)
            s = float(self.w2 @ h + self.b2[0])
            pred += s
            cache["z"].append(z)
            cache["h"].append(h)
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        T = cache["T"]
        codes = cache["codes"]
        ts = cache["ts"]
        seq = cache["seq"]
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros(1)
        dL_dq = [np.zeros(self.n_axes) for _ in range(T)]

        for i in range(T - 1):
            z, h = cache["z"][i], cache["h"][i]
            dL_ds = dL_dpred
            dw2 += dL_ds * h
            db2[0] += dL_ds
            dL_dh = dL_ds * self.w2
            dL_dpre = dL_dh * (1.0 - h ** 2)
            dW1 += np.outer(dL_dpre, z)
            db1 += dL_dpre
            dL_dz = self.W1.T @ dL_dpre
            dL_dq[i] += dL_dz[:self.n_axes]
            dL_dq[i + 1] += dL_dz[self.n_axes:]

        dWq_total = np.zeros_like(self.Wq)
        dbq_total = np.zeros_like(self.bq)
        for i in range(T):
            t = ts[i]
            dL_dt = dL_dq[i]                     # STE: identycznosc przez sign+balance_correct
            dL_dproj = dL_dt * (1.0 - t ** 2)
            dWq_total += np.outer(dL_dproj, seq[i])
            dbq_total += dL_dproj

        return {"Wq": dWq_total, "bq": dbq_total, "W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


def run_one(n_axes: int, seed: int, d_embed=8, seq_len=10, hidden=16,
            steps=400, batch_size=32, lr=0.02):
    ds_test = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=200 + seed)
    ds_ref = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed)
    ds_test.category_embed = ds_ref.category_embed.copy()

    rng = np.random.default_rng(seed)
    m = BottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng)
    train_khipu(m, ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100 + seed),
                steps=steps, batch_size=batch_size, lr=lr)
    mae = evaluate(m, ds_test, n_samples=500)
    n_params = sum(v.size for v in m.params().values())
    return mae, n_params


def run(dims=(4, 6, 9, 13, 16), seeds=(1, 2, 3), verbose=True, **kwargs):
    results = {d: [] for d in dims}
    n_params = {}
    for d in dims:
        for seed in seeds:
            mae, np_ = run_one(d, seed, **kwargs)
            results[d].append(mae)
            n_params[d] = np_
            if verbose:
                print(f"n_axes={d} seed={seed}: mae={mae:.4f}")
    if verbose:
        print()
        for d in dims:
            vals = results[d]
            print(f"n_axes={d} (params={n_params[d]}): mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    return results, n_params


if __name__ == "__main__":
    run()
