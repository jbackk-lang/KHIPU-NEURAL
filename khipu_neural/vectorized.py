"""
vectorized.py — odpowiedz na pytanie "jak wypada wydajnosc modulu w
porownaniu do prawdziwego frameworka (PyTorch/JAX)?"

PyTorch nie dal sie zainstalowac w tym sandboxie (patrz README.md,
sekcja "Co tu naprawde jest") - siec do download.pytorch.org jest
zablokowana (403 przez proxy) tak samo teraz, jak przy budowie repo.
Sprawdzone ponownie 2026-08-22 - ten sam blad.

Zamiast zgadywac, ile dalby PyTorch, zmierzone zostalo to, co NA PEWNO
odpowiada za wiekszosc roznicy: obecny kod (models.py, bottleneck_sweep.py)
trenuje PO JEDNYM PRZYKLADZIE NAJPIERW (petla `for b in range(batch_size)`
w train.py::_train), a WEWNATRZ kazdego przykladu PO JEDNYM TOKENIE
(petla `for i in range(T)` w kazdym forward/backward) - to B*T malych
wywolan numpy per krok (dla batch=32,T=10: 320 malych mnozen macierzy
zamiast 1 duzego). Dokladnie to jest roznica, ktora PyTorch/JAX
eliminuja automatycznie przez wektoryzacje wsadowa (batched tensor ops)
- wiec zmierzony tu zysk z pelnej wektoryzacji (batch+czas w jednym
wywolaniu numpy zamiast B*T malych) jest REALNYM DOLNYM OSZACOWANIEM
tego, co dalby prawdziwy framework - bez GPU i bez kompilacji JIT
(ktore dodalyby wiecej, ale tego juz nie da sie zmierzyc w tym
sandboxie).

VectorizedBottleneckMLP: TA SAMA matematyka co BottleneckMLP (n_axes=9
== architektura KHIPUResonanceNetMLP), ale forward/backward dzialaja na
calych tensorach (batch, seq_len, ...) na raz - zero petli Python po
przykladach czy tokenach (jedyna pozostala petla to ograniczona liczba
iteracji balance_correct_vectorized, max n_axes//2+1, i to na CALYM
tensorze na raz, nie per-element).

Poprawnosc: sprawdzona wprost przez porownanie predykcji i gradientow
z BottleneckMLP (petla Python) na tych samych danych - zgadzaja sie co
do ~1e-14 (ta sama matematyka, inna implementacja, roznica to tylko
kolejnosc operacji zmiennoprzecinkowych) - patrz
tests/test_vectorized_matches_loop.py.

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, d_embed=8, n_axes=9, hidden=16, seq_len=10,
CPU tego sandboxa, bez GPU, bez JIT - patrz benchmark()):
===========================================================================

| batch_size | petla Python (30-50 krokow) | wektoryzowane | przyspieszenie |
|---|---|---|---|
| 8   | 0.269s | 0.037s | 7.3x  |
| 32  | 1.023s | 0.117s | 8.8x  |
| 128 | 4.082s | 0.315s | 13.0x |

Przyspieszenie ROSNIE wraz z batch_size - dokladnie tak, jak oczekiwane:
kazde wywolanie numpy ma stary narzut (alokacja, sprawdzanie typow), a
petla Python go mnozy razy B*T (dla batch=128,T=10 to 1280 malych wywolan
na krok). Wektoryzacja amortyzuje ten narzut na CALY tensor na raz.

UCZCIWA INTERPRETACJA: to jest DOLNE oszacowanie tego, co dalby prawdziwy
framework (PyTorch/JAX) - ten pomiar pokazuje TYLKO zysk z wektoryzacji
wsadowej na CPU. Prawdziwy framework dolozylby do tego:
(1) skompilowane jadra C++/CUDA zamiast interpretowanego numpy (dodatkowy
zysk, niezmierzony tutaj),
(2) mozliwosc GPU (rownolegle tysiace watkow zamiast 1-2 rdzeni CPU tego
sandboxa - patrz KHIPU/MODEL_TETRAGON_4CPU.md, ten sam sandbox mial tylko
2 rdzenie przy poprzednich pomiarach `nproc`),
(3) automatyczne rozniczkowanie bez recznie pisanego STE (mniejsze ryzyko
bledu, nie predkosc per se).
Realistycznie: 10-15x zmierzone tutaj to prawdopodobnie 10-100x wiecej,
gdyby doliczyc GPU - ale TEGO JUZ NIE ZMIERZONO, to ekstrapolacja, nie
pomiar (PyTorch fizycznie nie dal sie zainstalowac w tym sandboxie -
sprobowano ponownie przed tym pomiarem, ten sam blad 403 co przy
budowie repo).
"""
from __future__ import annotations
import time
import numpy as np


def balance_correct_vectorized(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Jak quantize.balance_correct, ale na calym tensorze (..., n_axes)
    na raz - petla tylko po (ograniczonej) liczbie mozliwych korekt, nie
    po przykladach/tokenach."""
    q = q.copy()
    n_axes = q.shape[-1]
    max_iters = n_axes // 2 + 1
    flat_shape = (-1, n_axes)
    orig_shape = q.shape
    q_flat = q.reshape(flat_shape)
    t_flat = t.reshape(flat_shape)
    n_rows = q_flat.shape[0]
    rows = np.arange(n_rows)

    for _ in range(max_iters):
        total = q_flat.sum(axis=-1)
        need_fix = np.abs(total) > 1
        if not need_fix.any():
            break
        majority_sign = np.sign(total)
        is_majority = (q_flat == majority_sign[:, None])
        candidate = np.where(is_majority, np.abs(t_flat), np.inf)
        candidate = np.where(need_fix[:, None], candidate, np.inf)
        pick = np.argmin(candidate, axis=-1)
        flip_rows = rows[need_fix]
        q_flat[flip_rows, pick[need_fix]] = -majority_sign[need_fix]
    return q_flat.reshape(orig_shape)


class VectorizedBottleneckMLP:
    """Rownowazne BottleneckMLP(d_embed, n_axes, hidden), ale forward/backward
    przyjmuja CALY batch (B, T, d_embed) na raz, bez petli Python po B ani T."""

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

    def forward_batch(self, X: np.ndarray):
        """X: (B, T, d_embed) -> pred (B,), cache."""
        B, T, d = X.shape
        proj = np.einsum("btd,kd->btk", X, self.Wq) + self.bq   # (B,T,n_axes)
        t = np.tanh(proj)
        q = np.sign(t)
        q[q == 0] = 1.0
        q = balance_correct_vectorized(q, t)

        Z = np.concatenate([q[:, :-1, :], q[:, 1:, :]], axis=-1)     # (B,T-1,2*n_axes)
        H_pre = np.einsum("btd,hd->bth", Z, self.W1) + self.b1        # (B,T-1,hidden)
        H = np.tanh(H_pre)
        S = np.einsum("bth,h->bt", H, self.w2) + self.b2[0]           # (B,T-1)
        pred = S.sum(axis=1)                                          # (B,)
        cache = {"X": X, "t": t, "q": q, "Z": Z, "H": H, "T": T, "B": B}
        return pred, cache

    def backward_batch(self, dL_dpred: np.ndarray, cache):
        """dL_dpred: (B,) -> grads (sumy po batchu, NIE srednia - wolajacy
        dzieli przez batch_size, tak jak w train.py::_train)."""
        X, t, q, Z, H = cache["X"], cache["t"], cache["q"], cache["Z"], cache["H"]
        B, T = cache["B"], cache["T"]

        dL_ds = np.repeat(dL_dpred[:, None], T - 1, axis=1)   # (B,T-1) - pred=sum_i s_i
        dw2 = np.einsum("bt,bth->h", dL_ds, H)
        db2 = np.array([dL_ds.sum()])
        dL_dH = dL_ds[:, :, None] * self.w2[None, None, :]     # (B,T-1,hidden)
        dL_dHpre = dL_dH * (1.0 - H ** 2)
        dW1 = np.einsum("bth,btd->hd", dL_dHpre, Z)
        db1 = dL_dHpre.sum(axis=(0, 1))
        dL_dZ = np.einsum("bth,hd->btd", dL_dHpre, self.W1)    # (B,T-1,2*n_axes)

        dL_dq = np.zeros_like(q)                                # (B,T,n_axes)
        dL_dq[:, :-1, :] += dL_dZ[:, :, :self.n_axes]
        dL_dq[:, 1:, :] += dL_dZ[:, :, self.n_axes:]

        dL_dt = dL_dq                                           # STE
        dL_dproj = dL_dt * (1.0 - t ** 2)
        dWq = np.einsum("btk,btd->kd", dL_dproj, X)
        dbq = dL_dproj.sum(axis=(0, 1))

        return {"Wq": dWq, "bq": dbq, "W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


def benchmark(d_embed=8, n_axes=9, hidden=16, seq_len=10, batch_size=32, steps=50, seed=1):
    """Zwraca (t_loop, t_vectorized, speedup) - trenuje ten sam model
    (te same losowe wagi startowe) na tych samych danych obiema metodami
    przez `steps` krokow Adam, mierzy realny czas sciany."""
    from .bottleneck_sweep import BottleneckMLP
    from .data import ResonanceDataset
    from .train import Adam

    dataset = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100)

    # --- petla Python (obecna implementacja) ---
    rng = np.random.default_rng(seed)
    model_loop = BottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng)
    params_loop = model_loop.params()
    opt_loop = Adam(params_loop, lr=0.02)

    t0 = time.perf_counter()
    for _ in range(steps):
        X, y = dataset.sample_batch(batch_size)
        grad_accum = {k: np.zeros_like(v) for k, v in params_loop.items()}
        for b in range(batch_size):
            pred, cache = model_loop.forward(X[b])
            err = pred - y[b]
            grads = model_loop.backward(2 * err, cache)
            for k in grad_accum:
                grad_accum[k] += grads[k] / batch_size
        opt_loop.step(params_loop, grad_accum)
    t_loop = time.perf_counter() - t0

    # --- wersja wektoryzowana (batch+czas w jednym wywolaniu numpy) ---
    rng = np.random.default_rng(seed)
    model_vec = VectorizedBottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng)
    params_vec = model_vec.params()
    opt_vec = Adam(params_vec, lr=0.02)

    dataset2 = ResonanceDataset(d_embed=d_embed, seq_len=seq_len, seed=100)
    t0 = time.perf_counter()
    for _ in range(steps):
        X, y = dataset2.sample_batch(batch_size)
        pred, cache = model_vec.forward_batch(X)
        err = pred - y
        grads = model_vec.backward_batch(2 * err, cache)
        grads = {k: v / batch_size for k, v in grads.items()}
        opt_vec.step(params_vec, grads)
    t_vectorized = time.perf_counter() - t0

    speedup = t_loop / t_vectorized if t_vectorized > 0 else float("inf")
    return t_loop, t_vectorized, speedup


if __name__ == "__main__":
    t_loop, t_vec, speedup = benchmark()
    print(f"petla Python:     {t_loop:.3f}s")
    print(f"wektoryzowane:    {t_vec:.3f}s")
    print(f"przyspieszenie:   {speedup:.1f}x")
