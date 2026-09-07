# KHIPU-NEURAL

> **Status (2026-08-22): narzędzie o dobrze scharakteryzowanym, WĄSKIM
> zakresie stosowalności — nie "ogólnie lepsza architektura".** To repo
> testuje, czy koncepcja State9/GIPU z
> [jbackk-lang/KHIPU](https://github.com/jbackk-lang/KHIPU) daje się
> przełożyć na moduł sieci neuronowej i czy to cokolwiek daje. Po pełnej
> serii kontrolnych eksperymentów (poniżej) odpowiedź jest konkretna:
> **pomaga na zadaniach typu "wykryj zgodność/kategorię z zaszumionego
> sygnału ciągłego"** (dyskretyzacja działa jak wymuszone odszumianie),
> **i wyraźnie SZKODZI na zadaniach wymagających precyzyjnej wartości
> ciągłej** (dyskretyzacja niszczy dokładnie tę informację). Po drodze
> jeden pozorny "postęp" (13 osi bottlenecku > 9) okazał się artefaktem
> zbyt małej próby i został sam wykryty i cofnięty przez mechanizm
> samowalidacji (`self_validate.py`). Nic z tego nie zostało po drodze
> wyczyszczone ani ukryte — pełna historia (łącznie ze ślepymi
> zaułkami) jest w commitach i w sekcji "Wyniki" niżej.

## Skąd to się wzięło

[KHIPU](https://github.com/jbackk-lang/KHIPU) to deterministyczny,
symulowany procesor geometryczny (State9/F4-RED: 9-elementowy wektor
±1 z warunkiem równowagi, klasyfikacja skrętu/kierunku, relacje GIPU
między sąsiednimi węzłami). To repo to próba przełożenia tych zasad na
sieć neuronową (trening gradientowy) — **oddzielna od KHIPU**, bo to
inna domena (autodiff zamiast deterministycznej symulacji) i celowo nie
miesza się w jego trójwarstwową strukturę (kod / koncepcja / hipoteza).

## Co tu naprawdę jest (i czego nie ma)

PyTorch nie dał się zainstalować w tym środowisku (403 przez proxy do
`download.pytorch.org` — sprawdzone wielokrotnie, ostatnio przy okazji
`vectorized.py`, ten sam błąd za każdym razem) — więc **cały kod to
ręcznie napisany forward + backward w czystym NumPy**, bez frameworka
do automatycznego różniczkowania. Każda pochodna jest **zweryfikowana
numerycznym gradient-checkiem** (`tests/test_gradients_*.py`), nie
"na oko" — jeden z tych testów faktycznie złapał prawdziwy błąd we
własnym backpropie (subtelność STE przy iloczynie dwóch niezależnie
kwantowanych wektorów, `models.py::KHIPUResonanceNet.backward`).

## Architektura — wszystkie warianty

### Bottleneck: `State9Bottleneck` (`quantize.py`)

Ciągły wektor → uczona projekcja liniowa → `tanh` → twarda kwantyzacja
do ±1 na 9 osiach z korekcją równowagi (`|N+ - N-| <= 1`, ten sam
warunek co F4-RED w KHIPU) → gradient przez Straight-Through Estimator
(STE), jak w VQ-VAE. **Uczciwe zaszeregowanie**: koncepcyjnie to samo co
[Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)
(Mentzer i in., 2023) — jedyny unikalny element to kombinatoryczny
warunek równowagi (252 z 512 dozwolonych stanów), nieprzetestowany
gdzie indziej jako coś, co realnie pomaga.

### Modele w `models.py`

- **`BaselinePairwiseMLP`** — punkt odniesienia bez żadnej struktury
  KHIPU: zwykły 2-warstwowy MLP na konkatenacji sąsiadującej pary
  embeddingów.
- **`KHIPUResonanceNet`** — pierwsza, "dosłowna" wersja: kwantyzacja
  State9 + rezonans jako znormalizowany iloczyn skalarny dwóch kodów,
  zmieszany sigmoidem między dwoma uczonymi skalarami. **Przegrywa**
  nawet z trywialnym predyktorem (MAE ~1.08-1.15).
- **`KHIPUResonanceNetMLP`** — ta sama kwantyzacja, ale MLP zamiast
  dwóch skalarów jako agregacji. **Wygrywa** z baseline'em ~2x pod
  względem błędu — główny pozytywny wynik tego repo.
- **`KHIPUResonanceNetMLPNoQuant`** — ablacja: identyczny pipeline, ale
  BEZ twardej kwantyzacji (ciągłe `tanh` zamiast `sign`+korekty).
  Używana do izolowania efektu dyskretności od efektu bottlenecku.
- **`KHIPUResonanceNetStructured`** — próba usprawnienia: redukcja
  `q_i ⊙ q_i+1` (iloczyn per-oś, wprost reguła GIPU) przed MLP, zamiast
  surowej konkatenacji. **Nie wygrała jednoznacznie** — mniej
  parametrów, ale średnio gorszy wynik i większa wariancja.

### `bottleneck_sweep.py` — `BottleneckMLP`

Jak `KHIPUResonanceNetMLP`, ale z konfigurowalną liczbą osi bottlenecku
(nie na sztywno 9) — do testowania, czy 9 (wzięte z KHIPU, nie dobrane
pod to zadanie) jest w ogóle dobrym wyborem.

### `frozen_projection.py` — `FrozenBottleneckMLP`

Jak `BottleneckMLP`, ale projekcja `Wq/bq` jest losowa i **nigdy nie
uczona** — test, czy przewaga bottlenecku to tylko generyczna redukcja
wymiaru, czy realnie coś zależy od uczenia.

### `vectorized.py` — `VectorizedBottleneckMLP`

Ta sama matematyka co `BottleneckMLP`, ale forward/backward na całym
tensorze (batch, czas) naraz zamiast pętli Python po przykładach i
tokenach — zgodność z wersją pętlową zweryfikowana do ~1e-14.

### `self_validate.py` — samowalidacja jako proces samodoskonalenia

Dwie bramki przed każdą próbą zmiany domyślnej konfiguracji: (1)
gradient-check musi przejść, (2) wymiary macierzy muszą się zgadzać.
`self_improve_bottleneck()` dodatkowo wymaga minimum 5 ziaren losowości
i progu 10% względnej poprawy, zanim w ogóle rozważy zmianę — każda
decyzja zapisywana append-only do `self_improve_log.json`.

## Zadania testowe (`data.py`, `dotproduct_task.py`)

- **`ResonanceDataset`** — główne zadanie, zaprojektowane wprost pod
  regułę GIPU: każdy token ma ukrytą kategorię (nieznaną modelowi,
  widoczny tylko zaszumiony embedding), etykieta to liczba sąsiadujących
  par o tej samej kategorii. Korzystne dla bottlenecku z założenia.
- **`DotProductDataset`** — próba zadania niekorzystnego dla
  bottlenecku (ciągły iloczyn skalarny szumu). Okazała się **w ogóle
  nieuczalna** dla żadnego modelu (i baseline, i KHIPU wypadały jak
  zgadywanie średniej) — odrzucona jako wadliwy test, zastąpiona przez:
- **`DistanceDataset`** — te same ukryte kategorie co `ResonanceDataset`,
  ale etykieta to suma kwadratów odległości euklidesowych sąsiadów
  (ciągła, zależna od magnitudy, nie od dyskretnej zgodności). Zadanie
  faktycznie niekorzystne dla kwantyzacji — patrz "Wyniki".

## Wyniki (zmierzone, nie szacowane)

Wspólne ustawienia (chyba że zaznaczono inaczej): `d_embed=8`,
`seq_len=10`, 4 kategorie ukryte, 400 kroków, `batch=32`, Adam
(ręczny), `lr=0.02`.

**Główne porównanie** (`compare.py`, 3 ziarna; potwierdzone na 4
ziarnach w `ablation.py`):

| Model | Parametry | test MAE |
|---|---|---|
| BaselinePairwiseMLP | 289 | 0.253-0.286 |
| KHIPUResonanceNet (State9 + 2 skalary) | 83 | ~1.08-1.15 (przegrywa z trywialnym) |
| **KHIPUResonanceNetMLP** | 402 | **0.114-0.133** |
| trywialny (zawsze średnia) | 0 | ~1.03 |

**Kontrola przyczyny** (`ablation.py`, 4 ziarna) — czy to parametry czy
kwantyzacja?

| Wariant | Parametry | test MAE |
|---|---|---|
| Baseline dopasowany parametrami (hidden=22) | 397 | 0.260 ± 0.035 |
| KHIPUResonanceNetMLP | 402 | 0.133 ± 0.033 |
| KHIPUResonanceNetMLPNoQuant (bez kwantyzacji) | 402 | 0.142 ± 0.049 |

Więcej parametrów NIE tłumaczy wyniku (dopasowany baseline ledwo się
poprawia). Brak kwantyzacji wypada niemal identycznie jak z kwantyzacją
→ przewaga to głównie **redukcja wymiaru (bottleneck)**, nie
dyskretność sama w sobie.

**Sweep liczby osi bottlenecku** (`bottleneck_sweep.py`, najpierw 3
ziarna, potem `self_validate.py` dołożył 4 i 5) — **przykład
samokorekty**:

| n_axes | mean @ 3 ziarna | mean @ 5 ziaren |
|---|---|---|
| 9 (oryginalne, z State9/F4-RED) | 0.113 | **0.134 (zostaje)** |
| 13 | 0.100 ("najlepsze") | **0.159 (odwrócone, gorsze)** |

Na 3 ziarnach 13 osi wyglądało lepiej. Na 5 ziarnach — gorzej i mniej
stabilnie. `self_improve_bottleneck()` **poprawnie odmówił** promocji
na 13 właśnie przez wymóg min. 5 ziaren. Gdyby nie ten wymóg,
automatyczne "samodoskonalenie" popsułoby model. Pełna historia:
`self_improve_log.json`.

**Test: czy uczenie robi różnicę, czy to tylko losowa kompresja?**
(`frozen_projection.py`, 5 ziaren):

| Wariant | test MAE |
|---|---|
| BottleneckMLP (Wq/bq **uczone**) | 0.134 ± 0.029 |
| FrozenBottleneckMLP (Wq/bq **losowe, zamrożone**) | 0.552 ± 0.067 |

Zamrożona projekcja jest ~4x gorsza → uczenie realnie coś wnosi, to nie
jest tylko generyczna redukcja wymiaru.

**Test: czy przewaga generalizuje poza "zgodność kategorii"?**
(`dotproduct_task.py::DistanceDataset`, 5 ziaren):

| Model | test MAE |
|---|---|
| BaselinePairwiseMLP (ciągły) | 17.18 ± 6.37 |
| BottleneckMLP (kwantyzacja) | 24.54 ± 6.74 |
| trywialny (zawsze średnia) | 25.84 ± 5.18 |

Na zadaniu wymagającym ciągłej magnitudy kwantyzacja **ledwo bije
zgadywanie** i wyraźnie przegrywa z ciągłym baseline'em.

**Wydajność obliczeniowa vs. framework** (`vectorized.py`) — nie
dokładność, tylko szybkość: pętla Python (obecna implementacja) vs
pełna wektoryzacja wsadowa (batch+czas w jednym wywołaniu numpy),
identyczne wyniki (zgodność do ~1e-14):

| batch_size | pętla Python | wektoryzowane | przyspieszenie |
|---|---|---|---|
| 8 | 0.269s | 0.037s | 7.3x |
| 32 | 1.023s | 0.117s | 8.8x |
| 128 | 4.082s | 0.315s | 13.0x |

To dolne oszacowanie zysku z prawdziwego frameworka (PyTorch/JAX) —
skompilowane jądra i GPU dołożyłyby więcej, ale tego nie dało się
zmierzyć (PyTorch nie instaluje się w tym sandboxie).

## Wniosek końcowy: gdzie to się nadaje, a gdzie nie

Po całej serii testów (ablacja → sweep → samowalidacja → zamrożona
projekcja → zadanie magnitudowe) obraz jest spójny i **nie zmienia się
już jakościowo** przy dalszych testach tego samego rodzaju:

- **Nadaje się** do zadań, gdzie prawdziwy sygnał jest **kategorialny/
  dyskretny, ale obserwowany przez szum** — np. rozpoznawanie, czy dwie
  zaszumione obserwacje należą do tej samej (nieznanej) klasy. Twarda
  kwantyzacja działa tu jak wymuszone odszumianie: obcina niepewność
  PRZED porównaniem, zamiast każąc sieci uczyć się tego filtrowania
  samodzielnie.
- **Nie nadaje się** do zadań, gdzie liczy się **precyzyjna wartość
  ciągła / magnitude** — np. regresja odległości, sum ważonych,
  dowolna wielkość, gdzie strata informacji przy kwantyzacji do ±1 jest
  karana wprost. Tu kwantyzacja aktywnie szkodzi.
- **Kluczowy jest bottleneck (redukcja wymiaru), nie dyskretność sama w
  sobie** — ciągła wersja tego samego bottlenecku wypada niemal tak
  samo dobrze na zadaniu kategorialnym (patrz ablacja) i tak samo źle
  na zadaniu magnitudowym.
- **Liczba osi bottlenecku wymaga ostrożnego doboru z odpowiednią
  liczbą ziaren** — pozorne ulepszenia (13 zamiast 9) potrafią się
  odwrócić przy większej próbie.

### Możliwe zastosowania (w granicach powyższych ograniczeń)

Biorąc pod uwagę dokładnie to, co zostało zmierzone — nie więcej — ten
typ bottlenecku (uczona projekcja + twarda kwantyzacja z warunkiem
równowagi) ma sens tam, gdzie zadanie sprowadza się do **wykrywania
zgodności/przynależności do tej samej kategorii z zaszumionych,
ciągłych obserwacji**, a NIE do odtworzenia dokładnej wartości. Przykłady
o podobnym charakterze do przetestowanego zadania (spekulacja co do
przenoszalności, NIE zweryfikowana empirycznie poza `ResonanceDataset`):

- Wykrywanie duplikatów/dopasowań rekordów z zaszumionych embeddingów
  (np. czy dwa zaszumione odczyty czujnika pochodzą z tego samego
  źródła/reżimu), gdzie liczy się decyzja tak/nie, nie wartość.
  Analogiczne do reguły GIPU w KHIPU ("ten sam S i K → rezonans").
- Wstępny filtr/bottleneck przed klasyfikacją w warunkach silnego szumu
  pomiarowego — jako alternatywa dla zwykłego, ciągłego bottlenecku,
  do sprawdzenia w konkretnym przypadku (test z `frozen_projection.py`
  pokazuje, że MUSI być uczony, nie losowy).
- Kompresja stanu do dyskretnego "słownika" symboli o ograniczonej
  liczbie kombinacji (252 z 512 dla 9 osi) — potencjalnie przydatne,
  gdzie potrzebny jest dyskretny, interpretowalny kod pośredni, a nie
  tylko wynik końcowy.

Czego **NIE** warto próbować na podstawie tych wyników: regresji
wielkości ciągłych (odległości, sum, wag), zadań gdzie precyzja
numeryczna ponad kilka bitów ma znaczenie, ani ogólnego zastąpienia
zwykłego bottlenecku "bo działa lepiej" — bez własnego testu na
konkretnym zadaniu (metodologia z `ablation.py`/`frozen_projection.py`
jest do powtórzenia).

## Zastrzeżenia

- Wszystkie powyższe wyniki pochodzą z JEDNEJ rodziny syntetycznych
  zadań (`ResonanceDataset`/`DistanceDataset`, `d_embed=8`, 4 kategorie,
  `seq_len=10`) — nie z rzeczywistych danych. Kierunek wyników
  (kategorialne vs magnitudowe) jest logicznie umotywowany i spójny w
  obu testach, ale skala liczb (0.13 vs 0.29 MAE) jest specyficzna dla
  tej konfiguracji.
- Trening w ręcznym NumPy (bez wektoryzacji) jest wolniejszy niż
  konieczne — patrz `vectorized.py` dla zmierzonego, dolnego
  oszacowania zysku z wektoryzacji (7-13x).
- Poziom szumu (`noise_std=0.4`) i liczba kategorii (4) nie były
  systematycznie przeskanowane — możliwe, że granica
  "kategorialne vs magnitudowe" jest płynna, nie ostra, w zależności od
  tych parametrów. Nie sprawdzone.

## Uruchomienie

```bash
pip install -r requirements.txt
pytest tests/ -v                        # 25 testow
python3 -m khipu_neural.compare          # glowne porownanie, ~35s
python3 -m khipu_neural.ablation         # kontrola parametry/kwantyzacja, ~3-4 min
python3 -m khipu_neural.bottleneck_sweep # sweep liczby osi, ~5 min
python3 -m khipu_neural.self_validate    # samowalidacja + samodoskonalenie
python3 -m khipu_neural.vectorized       # benchmark predkosci petla vs wektoryzacja
```

## Testy

25 testów w `tests/`:
- `test_gradients_quantize.py` — `State9Bottleneck`, gradienty do ~1e-4/1e-11.
- `test_gradients_models.py` — gradienty wszystkich modeli, w tym
  poprawnie zlinearyzowany STE przez iloczyn dwóch skwantyzowanych
  wektorów (test, który złapał prawdziwy błąd w historii gita).
- `test_training_smoke.py` — pętla treningowa zmniejsza stratę, brak NaN/inf.
- `test_ablation_smoke.py` — kontrola parametry/kwantyzacja działa end-to-end.
- `test_bottleneck_sweep_smoke.py` — parametryzowany bottleneck, gradienty do ~1e-11.
- `test_self_validate_smoke.py` — obie bramki + append-only log samodoskonalenia.
- `test_vectorized_matches_loop.py` — wektoryzacja zgodna z pętlą do ~1e-14.
- `test_frozen_and_distance_smoke.py` — zamrożona projekcja + zadanie magnitudowe.

## Licencja

MIT — patrz [LICENSE](LICENSE).
