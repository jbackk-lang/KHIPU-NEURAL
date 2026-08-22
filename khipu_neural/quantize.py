"""
quantize.py — State9Bottleneck: rozniczkowalna (uczona) wersja State9/F4-RED
z KHIPU (khipu/node256.py, khipu/cpu.py w repo jbackk-lang/KHIPU).

W KHIPU: word16 -> (recznie zaprojektowana heurystyka popcount) -> S (jedna
z 7 klas skretu), bez uczenia. Tutaj: ciagly wektor embeddingu -> uczona
projekcja liniowa -> tanh -> twarda kwantyzacja do +/-1 na 9 osiach z
przepustem gradientu (Straight-Through Estimator, jak w VQ-VAE) -> korekta
rownowagi (ten sam warunek |N+ - N-| <= 1, co filtr F4-RED w KHIPU),
rowniez z STE.

To jest RECZNIE napisany forward+backward (bez PyTorch/autograd - w tym
sandboxie nie dalo sie pobrac torcha, sieci PyPI byla zbyt wolna/zablokowana).
Kazda funkcja backward jest zweryfikowana numerycznym gradient-checkiem
w tests/test_gradients.py - to NIE jest "na oko", tylko sprawdzone.
"""
from __future__ import annotations
import numpy as np

N_AXES = 9  # jak w State9 (KHIPU: khipu/node256.py, F4-RED)


def balance_correct(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Koryguje wektor +/-1 `q` (dlugosc 9) tak, zeby |sum(q)| <= 1 (warunek
    F4-RED: |N+ - N-| <= 1). Jesli jest niezgodny, odwraca element
    wiekszosciowej grupy o NAJMNIEJSZEJ |t| (najblizszy granicy decyzyjnej,
    "najmniej pewny") - ten sam duch co JCompressor._reduce_arm w KHIPU:
    minimalna ingerencja, najpierw tam gdzie jest najmniej pewności.

    Dziala na pojedynczym wektorze (9,). Deterministyczne.
    """
    q = q.copy()
    total = int(q.sum())
    while abs(total) > 1:
        majority_sign = 1 if total > 0 else -1
        # indeksy nalezace do wiekszosciowej grupy
        idx = np.where(q == majority_sign)[0]
        # najmniej pewny (najmniejsza |t|) z tej grupy
        pick = idx[np.argmin(np.abs(t[idx]))]
        q[pick] = -majority_sign
        total = int(q.sum())
    return q


class State9Bottleneck:
    """
    Warstwa: x (d,) -> proj = Wq@x + bq (9,) -> t = tanh(proj) -> q = sign(t)
    -> q_corrected = balance_correct(q, t)  (9,), wartosci +/-1, |sum|<=1.

    Backward: STE na sign() I na balance_correct() - gradient przechodzi
    tak, jakby q_corrected == t (tozsamosc), az do tanh (ktore MA prawdziwa
    pochodna 1-tanh^2). To standardowy STE, tak jak w VQ-VAE/FSQ.
    """

    def __init__(self, d_in: int, rng: np.random.Generator):
        scale = 1.0 / np.sqrt(d_in)
        self.Wq = rng.normal(0, scale, size=(N_AXES, d_in))
        self.bq = np.zeros(N_AXES)
        self._cache = {}

    def params(self):
        return {"Wq": self.Wq, "bq": self.bq}

    def forward(self, x: np.ndarray) -> np.ndarray:
        proj = self.Wq @ x + self.bq          # (9,)
        t = np.tanh(proj)                      # (9,)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct(q, t)
        self._cache = {"x": x, "proj": proj, "t": t}
        return q

    def backward(self, dL_dq: np.ndarray) -> tuple[np.ndarray, dict]:
        """dL_dq: gradient wzgledem wyjscia (9,) - STE: traktujemy jakby
        q == t (tozsamosc). Zwraca (dL_dx, {dWq, dbq})."""
        t = self._cache["t"]
        x = self._cache["x"]
        dL_dt = dL_dq                                  # STE przez sign+korekte
        dL_dproj = dL_dt * (1.0 - t ** 2)               # pochodna tanh
        dWq = np.outer(dL_dproj, x)
        dbq = dL_dproj
        dL_dx = self.Wq.T @ dL_dproj
        return dL_dx, {"Wq": dWq, "bq": dbq}
