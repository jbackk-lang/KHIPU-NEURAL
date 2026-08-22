# KHIPU-NEURAL

> **Status: eksperymentalny prototyp badawczy, NIE stan sztuki.** To repo
> testuje, czy koncepcje geometryczne z
> [jbackk-lang/KHIPU](https://github.com/jbackk-lang/KHIPU) dają się
> przełożyć na moduł sieci neuronowej i czy pomagają. **Wynik jest
> mieszany, ciekawszy niż prosto pozytywny/negatywny**: pierwsza,
> "dosłowna" wersja (dwa skalary jako agregacja) przegrywa nawet z
> trywialnym predyktorem; druga wersja (ten sam bottleneck State9, ale
> z MLP zamiast dwóch skalarów) **konsekwentnie WYGRYWA** z czysto
> ciągłym baseline'em, ~2x pod względem błędu, powtarzalnie na 3
> ziarnach losowości. Obie wersje i cała droga dojścia do tego wniosku
> są w tym repo — nic nie zostało po drodze wyczyszczone czy ukryte.

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

### `KHIPUResonanceNetMLP` (`models.py`)

Ta sama kwantyzacja State9 co wyżej, ale zamiast dwóch skalarów jako
agregacji — mały 2-warstwowy MLP nad konkatenacją `[q_i, q_i+1]`
(18-wymiarowy, dyskretny wektor ±1), tej samej wielkości ukrytej co
`BaselinePairwiseMLP`. To wariant, który **wygrywa** — patrz "Wyniki".

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
`batch=32`, Adam (ręczny), `lr=0.02`. Test MAE uśredniony z 3 niezależnych
ziaren losowości (tam gdzie sprawdzone wielokrotnie).

| Model | Parametry | test MAE | czas treningu |
|---|---|---|---|
| BaselinePairwiseMLP (baseline, ciągły) | 289 | 0.253 ± 0.032 | 5.4s |
| KHIPUResonanceNet (State9 + **dwa skalary**) | 83 | ~1.08 (1 przebieg) | 26.0s |
| **KHIPUResonanceNetMLP** (State9 + **MLP**) | 402 | **0.114 ± 0.014** | 18.9s |
| trywialny (zawsze średnia) | 0 | 1.03 | — |

Pojedyncze ziarna dla `KHIPUResonanceNetMLP` vs baseline (ta sama trójka
ziaren, ten sam podział train/test): 0.098 vs 0.218, 0.133 vs 0.247,
0.111 vs 0.295 — **KHIPU z MLP wygrywa na każdym z 3 ziaren**, nie tylko
średnio.

Próba strojenia pierwszej wersji (`KHIPUResonanceNet`, niższy `lr=0.005`,
temperatura sigmoid 2.0 zamiast 4.0, 800 kroków zamiast 400) dała test
MAE = 1.15 — gorzej, nie lepiej. Problemem NIE był brak strojenia.

### Co się zmieniło i dlaczego to ważne

Pierwsza wersja (`KHIPUResonanceNet`) łączyła dwa niezależne pomysły:
(1) twardą kwantyzację State9 i (2) skrajnie ubogą agregację (tylko dwa
uczone skalary, `w_resonant`/`w_other`, wybierane przez jedną bramkę
sigmoid). Porażka mogła wynikać z każdego z nich osobno albo z obu.
`KHIPUResonanceNetMLP` zmienia WYŁĄCZNIE (2) — zamiast dwóch skalarów,
mały 2-warstwowy MLP nad konkatenacją dwóch kodów State9 (tej samej
wielkości ukrytej co baseline). To rozdziela zmienne: skoro po tej
jednej zmianie architektura nie tylko przestaje przegrywać, ale zaczyna
wyraźnie wygrywać, to **dyskretna kwantyzacja State9 sama w sobie nie
była problemem — problemem był zbyt ubogi sposób jej wykorzystania.**

### Możliwe wyjaśnienie przewagi

Dane wejściowe to zaszumione obserwacje niewielkiej liczby (4) ukrytych
kategorii. Twarda kwantyzacja do dyskretnego kodu działa jak wymuszona
decyzja kategoryczna, usuwająca szum PRZED porównaniem sąsiadów —
ciągły baseline musi sam nauczyć się odfiltrować ten szum wewnątrz MLP,
co jest trudniejszym zadaniem uczenia się. To spójne z tym, dlaczego
kwantyzacja (VQ-VAE i pokrewne) bywa używana jako mechanizm
odszumiający/regularyzujący gdzie indziej — tutaj widać to samo zjawisko
na małą skalę.

### Zastrzeżenia (żeby nie przereklamować)

- To jedno małe, syntetyczne zadanie zaprojektowane wprost pod regułę
  GIPU — nie dowód ogólnej wyższości tego podejścia.
- `KHIPUResonanceNetMLP` ma więcej parametrów (402) niż baseline (289)
  — część przewagi MOŻE częściowo wynikać z większej pojemności, nie
  tylko z samej kwantyzacji. Uczciwe porównanie "parametr do parametru"
  (baseline z większą warstwą ukrytą, dopasowaną liczbą parametrów) nie
  zostało tu jeszcze sprawdzone — naturalny następny krok.
- Trening w ręcznym NumPy jest ~3-5x wolniejszy niż baseline z powodu
  narzutu STE i pętli po tokenach w czystym Pythonie — realna
  implementacja w PyTorch/JAX z wektoryzacją byłaby szybsza, nie tylko
  bardziej wygodna.

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
