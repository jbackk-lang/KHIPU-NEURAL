# KHIPU-NEURAL

> **Status: eksperymentalny prototyp badawczy, NIE stan sztuki.** To repo
> testuje jedną konkretną hipotezę (czy koncepcje geometryczne z
> [jbackk-lang/KHIPU](https://github.com/jbackk-lang/KHIPU) dają się
> przełożyć na moduł sieci neuronowej i czy pomagają) — i **na
> przetestowanym zadaniu wynik jest negatywny** (patrz "Wyniki" niżej).
> To nie jest ukrywane ani upiększane — cały sens tego repo to uczciwy
> pomiar, nie promocja pomysłu.

## Skąd to się wzięło

[KHIPU](https://github.com/jbackk-lang/KHIPU) to deterministyczny,
symulowany procesor geometryczny (State9/F4-RED: 9-elementowy wektor
±1 z warunkiem równowagi, klasyfikacja skrętu/kierunku, relacje GIPU
między sąsiednimi węzłami). Podczas rozmowy o tym repo padło pytanie: co
by było, gdyby te same zasady spróbować wbudować w sieć neuronową
zamiast w deterministyczną symulację? To repo to właśnie ta próba —
**oddzielne od KHIPU**, bo używa innej domeny (trening gradientowy,
PyTorch-podobne API, inna kategoria zależności) niż czysto deterministyczna
symulacja w KHIPU, i celowo nie miesza się w jego uporządkowaną,
trójwarstwową strukturę (kod / koncepcja / hipoteza).

## Co tu naprawdę jest (i czego nie ma)

PyTorch nie dał się zainstalować w środowisku, w którym to powstawało
(sieć zbyt wolna/zablokowana do pobrania pakietu) — więc **cały kod to
ręcznie napisany forward + backward w czystym NumPy**, bez żadnego
frameworka do automatycznego różniczkowania. To ważne zastrzeżenie:
każda pochodna w tym repo jest wyprowadzona i zaimplementowana ręcznie,
i **każda jest zweryfikowana numerycznym gradient-checkiem**
(`tests/test_gradients_*.py`) — nie "na oko". Przy pisaniu jednego z
tych testów gradient-check faktycznie złapał prawdziwy błąd w moim
własnym ręcznym backpropie (subtelność STE przy iloczynie dwóch
niezależnie kwantowanych wektorów) — poprawiony i udokumentowany wprost
w kodzie (`khipu_neural/models.py`, docstring `KHIPUResonanceNet.backward`).

## Architektura

### `State9Bottleneck` (`quantize.py`)

Rozniczkowalna wersja State9/F4-RED z KHIPU: ciągły wektor →
uczona projekcja liniowa → `tanh` → twarda kwantyzacja do ±1 na 9 osiach
z korekcją równowagi (`|N+ - N-| <= 1`, dokładnie ten sam warunek co
filtr F4-RED w KHIPU) → gradient przechodzi przez Straight-Through
Estimator (STE), jak w VQ-VAE.

**Uczciwe zaszeregowanie w istniejącej literaturze**: to NIE jest
nowy pomysł na dyskretną kwantyzację — [Finite Scalar Quantization
(Mentzer i in., 2023)](https://arxiv.org/abs/2309.15505) robi
koncepcyjnie to samo (kwantyzacja niezależna po osi, bez uczonego
codebooka jak w VQ-VAE). Jedyne, co State9 dokłada ponad FSQ, to
**kombinatoryczny warunek równowagi** ograniczający dozwolone
kombinacje (252 z 512) — to jest realny, unikalny element, ale nie
sprawdzony empirycznie gdzie indziej ani tutaj jako coś, co pomaga.

### `KHIPUResonanceNet` (`models.py`)

Każdy token jest kwantowany przez `State9Bottleneck`, potem "rezonans"
między sąsiadami liczony jest jako znormalizowany iloczyn skalarny
dwóch kodów ±1 (miękka, różniczkowalna wersja reguły GIPU: "ten sam S
i K → rezonans" z `khipu/gipu.py` w KHIPU), przepuszczony przez sigmoid
jako bramka mieszająca dwie uczone wartości skalarne
(`w_resonant`, `w_other`).

### `BaselinePairwiseMLP` (`models.py`)

Punkt odniesienia bez żadnej struktury KHIPU: zwykły 2-warstwowy MLP na
konkatenacji sąsiadującej pary embeddingów.

## Zadanie testowe: "liczenie rezonansów" (`data.py`)

Zaprojektowane **wprost pod regułę GIPU**, żeby dać architekturze
KHIPU najlepszą możliwą szansę: każdy token ma ukrytą kategorię
(nieznaną modelowi — widoczny jest tylko zaszumiony ciągły embedding),
etykieta to liczba sąsiadujących par o tej samej ukrytej kategorii
("par rezonansowych"). To dokładnie zadanie, do którego GIPU zostało
zaprojektowane w KHIPU (tam: reguła ręczna, tu: ucząca się).

## Wyniki (zmierzone, nie szacowane)

Ustawienia: `d_embed=8`, `seq_len=10`, 4 kategorie ukryte, 400 kroków,
`batch=32`, Adam (ręczny), `lr=0.02`.

| Model | Parametry | MSE koniec | test MAE | czas treningu |
|---|---|---|---|---|
| BaselinePairwiseMLP | 289 | 0.13 | **0.365** | 5.4s |
| KHIPUResonanceNet | 83 | 1.52 (niestabilne) | 1.08 | 26.0s (~5x wolniej) |
| trywialny (zawsze średnia) | 0 | — | 1.03 | — |

Próba strojenia KHIPU (niższy `lr=0.005`, temperatura sigmoid 2.0
zamiast 4.0, 800 kroków zamiast 400) dała test MAE = 1.15 — **gorzej**,
nie lepiej.

### Uczciwy wniosek

Generyczny MLP bez żadnej "geometrycznej" struktury uczy się
zdecydowanie lepiej, szybciej i stabilniej niż architektura inspirowana
KHIPU — **nawet na zadaniu zaprojektowanym pod dokładnie tę regułę,
którą KHIPU miało realizować**. `KHIPUResonanceNet` w praktyce nie bije
trywialnego predyktora średniej.

To NIE jest dowód, że taka architektura nigdy nie może zadziałać w
żadnej formie. Prawdopodobne przyczyny porażki akurat tej
implementacji:

1. **Ostateczna agregacja to tylko 2 skalary** (`w_resonant`, `w_other`)
   — dużo mniejsza pojemność funkcyjna niż pełny MLP z 16 neuronami
   ukrytymi. Uczciwsze porównanie wymagałoby MLP o podobnej liczbie
   parametrów jako głowicy nad kodami State9, nie dwóch skalarów.
2. **Twarda kwantyzacja + STE jest z natury trudniejsza do
   optymalizacji** niż gładka reprezentacja ciągła — widać to we
   niestabilnej krzywej MSE (skok w górę w połowie treningu), typowy
   objaw dla STE bez dodatkowych sztuczek (np. temperature annealing,
   commitment loss jak w VQ-VAE, którego tu nie ma).
3. Możliwe, że przy dłuższym treningu/lepszym strojeniu (nie
   sprawdzonym tutaj wyczerpująco) wynik by się poprawił — ale to już
   wymagałoby realnego frameworka (PyTorch/JAX) do rozsądnie szybkiej
   iteracji, nie ręcznego NumPy.

## Uruchomienie

```bash
pip install -r requirements.txt
pytest tests/ -v            # 9 testów: gradient-checki + trening dymny
python3 -m khipu_neural.compare   # pelny eksperyment, ~35s
```

## Testy

9 testów w `tests/`:
- `test_gradients_quantize.py` — `State9Bottleneck` zawsze produkuje
  poprawny stan F4-RED (`|suma| <= 1`), gradienty zweryfikowane
  numerycznie do ~1e-4/1e-11.
- `test_gradients_models.py` — gradienty obu modeli (w tym poprawnie
  zlinearyzowany numeryczny gradient przez STE na iloczynie dwóch
  skwantyzowanych wektorów — to jest test, który złapał prawdziwy błąd
  we wcześniejszej wersji `backward()`, patrz historia gita).
- `test_training_smoke.py` — pętla treningowa faktycznie zmniejsza
  stratę (baseline) i nie produkuje NaN/inf (khipu).

## Licencja

MIT — patrz [LICENSE](LICENSE).
