"""
frozen_projection.py — najostrzejszy test tego, CZY cokolwiek specyficznego
dla KHIPU/State9/GIPU odpowiada za przewage bottlenecku, czy to tylko
zwykla redukcja wymiaru (jak losowy rzut Johnson-Lindenstrauss).

FrozenBottleneckMLP: DOKLADNIE ta sama architektura co BottleneckMLP
(n_axes=9, ten sam State9-jak bottleneck + MLP), ale Wq/bq sa LOSOWE I
NIGDY NIE UCZONE (nie ma ich w params(), wiec Adam ich nie dotyka) -
uczy sie WYLACZNIE glowa MLP (W1,b1,w2,b2) nad losowa, stala projekcja
9-osiowa + kwantyzacja State9.

Jesli FrozenBottleneckMLP wypada PODOBNIE do w pelni uczonego
BottleneckMLP -> przewaga to ZWYKLA redukcja wymiaru do 9, "geometria
State9" (uczona projekcja Wq, kwantyzacja, warunek rownowagi) NIE
DOKLADA nic ponad to. Jesli wypada WYRAZNIE gorzej -> uczenie Wq
faktycznie cos wnosi, warto drazyc dalej (inne zadania, generalizacja).

===========================================================================
ZMIERZONE WYNIKI (2026-08-22, 5 ziaren, ResonanceDataset, d_embed=8,
n_axes=9, hidden=16, 400 krokow):
===========================================================================

| wariant                          | test MAE (srednia +/- std) |
|---|---|
| BottleneckMLP (Wq/bq UCZONE)      | 0.134 +/- 0.029             |
| FrozenBottleneckMLP (Wq/bq LOSOWE, zamrozone) | 0.552 +/- 0.067 |

WYNIK JEDNOZNACZNY: zamrozona/losowa projekcja jest ~4x GORSZA niz
uczona. To OBALA hipoteze "to tylko zwykla redukcja wymiaru" - uczenie
Wq faktycznie robi cos istotnego (dopasowuje projekcje do struktury
ukrytych kategorii w danych), nie tylko losowo kompresuje sygnal.
WNIOSEK: warto bylo isc dalej - patrz dotproduct_task.py (test 2:
CZY ta przewaga jest waska/specyficzna dla zadan typu "zgodnosc
kategorii", czy szersza).
"""
from __future__ import annotations
from .bottleneck_sweep import BottleneckMLP


class FrozenBottleneckMLP(BottleneckMLP):
    """Jak BottleneckMLP, ale Wq/bq sa losowe i NIGDY nie uczone (nie
    wystepuja w params(), wiec Adam ich nie aktualizuje). Uczy sie
    wylacznie MLP nad stala, losowa projekcja+kwantyzacja."""

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2}
