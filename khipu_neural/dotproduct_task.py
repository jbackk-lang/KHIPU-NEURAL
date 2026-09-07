"""
dotproduct_task.py — Test 2: zadanie NIEKORZYSTNE dla bottlenecku, zeby
sprawdzic, czy przewaga z data.py::ResonanceDataset jest waska (dziala
tylko, gdy prawdziwa struktura jest kategorialna/dyskretna) czy szersza.

DotProductDataset: y = suma_i (x_i . x_i+1) (iloczyn skalarny sasiadow,
CIAGLA wielkosc zalezna od DOKLADNEJ MAGNITUDY wektorow). Brak ukrytych
kategorii - to CZYSTA regresja na surowym sygnale ciaglym. Twarda
kwantyzacja do +/-1 niszczy niemal cala informacje o magnitudzie
potrzebna do tego zadania - oczekiwanie: tu bottleneck+kwantyzacja
POWINIEN przegrac z ciaglym baseline, jesli wczesniejsza przewaga
faktycznie byla specyficzna dla zadan typu "zgodnosc kategorii".
"""
from __future__ import annotations
import numpy as np


class DotProductDataset:
    def __init__(self, d_embed: int = 8, seq_len: int = 10, seed: int = 0):
        self.d_embed = d_embed
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size: int):
        X = self._rng.normal(0, 1.0, size=(batch_size, self.seq_len, self.d_embed))
        y = np.einsum("btd,btd->bt", X[:, :-1, :], X[:, 1:, :]).sum(axis=1)
        return X, y


class DistanceDataset:
    """POPRAWIONA wersja zadania-testu: DotProductDataset okazal sie
    NIEUCZALNY (kazdy token to swiezy, niepowtarzalny szum bez zadnej
    struktury do wyuczenia - ani baseline, ani khipu nie pobily
    trywialnego predyktora, patrz frozen_projection.py/README). Tutaj,
    tak jak w data.py::ResonanceDataset, jest STALY slownik ukrytych
    kategorii uzywany wielokrotnie w calym datasetcie (wiec jest co
    uczyc), ale etykieta jest CIAGLA i zalezna od MAGNITUDY (suma
    kwadratow odleglosci euklidesowych miedzy sasiadami), nie od
    dyskretnej zgodnosci kategorii - to ma faktycznie utrudnic zycie
    twardej kwantyzacji +/-1 (ktora niszczy informacje o odlegowsci),
    zeby sprawdzic, czy przewaga bottlenecku jest specyficzna dla
    zadan typu "ta sama kategoria tak/nie".

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, 5 ziaren, DistanceDataset, d_embed=8,
n_axes=9, hidden=16, 400 krokow; trywialny = zawsze srednia z treningu,
liczony OSOBNO per ziarno bo category_embed jest losowe za kazdym razem):
===========================================================================

| model                          | test MAE (srednia +/- std) |
|---|---|
| BaselinePairwiseMLP (ciagly)    | 17.18 +/- 6.37   (WYRAZNIE uczy sie) |
| BottleneckMLP (n_axes=9, kwantyzacja) | 24.54 +/- 6.74  (ledwo lepszy niz zgadywanie) |
| trywialny (zawsze srednia)      | 25.84 +/- 5.18   |

WYNIK JEDNOZNACZNY (i zgodny z oczekiwaniem): na zadaniu wymagajacym
CIAGLEJ magnitudy (suma kwadratow odleglosci euklidesowych sasiadow,
nie dyskretnej zgodnosci kategorii), kwantyzacja do +/-1 NISZCZY
informacje potrzebna do dobrej predykcji - bottleneck ledwo bije
zgadywanie sredniej, podczas gdy ciagly baseline uczy sie realnej
struktury (17.2 vs 25.8).

WNIOSEK KONCOWY (test 1 + test 2 razem): przewaga State9-bottlenecku
NIE jest ani (a) samym zamrozonym/losowym rzutem [test 1 to obalil -
uczenie robi realna roznice], ANI (b) uniwersalnym usprawnieniem
niezaleznym od zadania [test 2 to obala - tutaj przegrywa nawet z
trywialnym predyktorem]. To NARZEDZIE WASKIEGO ZASTOSOWANIA: pomaga
na zadaniach typu "wykryj zgodnosc/rownosc kategorii z zaszumionego
sygnalu" (bo dyskretyzacja dziala jak wymuszona decyzja, filtrujac
szum), i AKTYWNIE SZKODZI na zadaniach wymagajacych precyzyjnej
ciaglej magnitudy (bo dyskretyzacja niszczy dokladnie te informacje).
Nie ma tu "ogolnej sprawnosci" do odkrycia - jest dobrze okreslony
zakres stosowalnosci, i tyle."""

    def __init__(self, d_embed: int = 8, n_categories: int = 4,
                 seq_len: int = 10, noise_std: float = 0.4, seed: int = 0):
        self.d_embed = d_embed
        self.n_categories = n_categories
        self.seq_len = seq_len
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        self.category_embed = rng.normal(0, 1.0, size=(n_categories, d_embed))
        self._rng = rng

    def sample_batch(self, batch_size: int):
        X = np.zeros((batch_size, self.seq_len, self.d_embed))
        y = np.zeros(batch_size, dtype=np.float64)
        for b in range(batch_size):
            cats = self._rng.integers(0, self.n_categories, size=self.seq_len)
            noise = self._rng.normal(0, self.noise_std, size=(self.seq_len, self.d_embed))
            X[b] = self.category_embed[cats] + noise
            true_embed = self.category_embed[cats]  # PRAWDZIWA (bez szumu) pozycja - to model musi ocenic
            diffs = true_embed[:-1] - true_embed[1:]
            y[b] = float(np.sum(diffs ** 2))  # suma kwadratow odleglosci euklidesowych sasiadow
        return X, y
