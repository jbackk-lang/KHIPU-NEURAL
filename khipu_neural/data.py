"""
data.py — syntetyczne zadanie "liczenie rezonansow": test wprost
odpowiadajacy regule GIPU z KHIPU (ten sam S i K miedzy sasiadami ->
rezonans, khipu/gipu.py: `relation_between`).

Kazdy token ma ukryta kategorie c_i (nieznana modelowi), obserwowany
jest tylko zaszumiony embedding x_i = E[c_i] + szum. Etykieta y = liczba
sasiadujacych par (i, i+1) o tej samej ukrytej kategorii ("par
rezonansowych"). Model musi nauczyc sie wykrywac zgodnosc ukrytej
kategorii miedzy sasiadami z samego zaszumionego sygnalu ciaglego -
dokladnie to zadanie, do ktorego GIPU/relation_between zostalo
zaprojektowane w KHIPU (tam: regula reczna, tu: uczona).
"""
from __future__ import annotations
import numpy as np


class ResonanceDataset:
    def __init__(self, d_embed: int = 8, n_categories: int = 4,
                 seq_len: int = 10, noise_std: float = 0.4, seed: int = 0):
        self.d_embed = d_embed
        self.n_categories = n_categories
        self.seq_len = seq_len
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        # stale, losowe embeddingi kategorii (nieznane modelowi - model widzi tylko x_i)
        self.category_embed = rng.normal(0, 1.0, size=(n_categories, d_embed))
        self._rng = rng

    def sample_batch(self, batch_size: int):
        """Zwraca (X, y): X to lista batch_size sekwencji (seq_len, d_embed),
        y to wektor (batch_size,) liczby par rezonansowych (int)."""
        X = np.zeros((batch_size, self.seq_len, self.d_embed))
        y = np.zeros(batch_size, dtype=np.float64)
        for b in range(batch_size):
            cats = self._rng.integers(0, self.n_categories, size=self.seq_len)
            noise = self._rng.normal(0, self.noise_std, size=(self.seq_len, self.d_embed))
            X[b] = self.category_embed[cats] + noise
            y[b] = float(np.sum(cats[:-1] == cats[1:]))
        return X, y
