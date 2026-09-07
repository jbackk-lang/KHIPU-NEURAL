"""
models.py — dwa modele do porownania na zadaniu "liczenie rezonansow"
(data.py):

  BaselinePairwiseMLP  — generyczny model bez struktury KHIPU: kazda
                          sasiadujaca para (x_i, x_i+1) idzie razem przez
                          zwykly 2-warstwowy MLP, ktory sam musi sie
                          nauczyc, czego szukac.

  KHIPUResonanceNet     — kazdy token jest najpierw kwantowany do State9
                          (quantize.State9Bottleneck, uczona wersja
                          F4-RED z KHIPU), potem "rezonans" miedzy
                          sasiadami liczony jest jako znormalizowany
                          iloczyn skalarny kodow +/-1 (miekka, rozniczkowalna
                          wersja reguly GIPU: "ten sam S i K -> rezonans"),
                          przepuszczony przez sigmoid jako bramka [0,1]
                          mieszajaca dwie uczone wartosci (w_rezonans,
                          w_inne).

Oba modele: reczny forward+backward w NumPy (bez frameworka - patrz
quantize.py), zweryfikowane numerycznym gradient-checkiem w
tests/test_gradients_models.py.
"""
from __future__ import annotations
import numpy as np
from .quantize import State9Bottleneck, N_AXES


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BaselinePairwiseMLP:
    def __init__(self, d_embed: int, hidden: int, rng: np.random.Generator):
        d_in = 2 * d_embed
        scale1 = 1.0 / np.sqrt(d_in)
        scale2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, scale1, size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, size=hidden)
        self.b2 = np.zeros(1)  # tablica 1-elem., nie float - zeby Adam mial prawdziwa referencje
        self.d_embed = d_embed

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def forward(self, seq: np.ndarray):
        """seq: (T, d_embed) -> pred (float), cache do backward."""
        T = seq.shape[0]
        cache = {"z": [], "h": []}
        pred = 0.0
        for i in range(T - 1):
            z = np.concatenate([seq[i], seq[i + 1]])   # (2d,)
            h = np.tanh(self.W1 @ z + self.b1)          # (hidden,)
            s = float(self.w2 @ h + self.b2[0])
            pred += s
            cache["z"].append(z)
            cache["h"].append(h)
        cache["T"] = T
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        T = cache["T"]
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros(1)
        for i in range(T - 1):
            z, h = cache["z"][i], cache["h"][i]
            dL_ds = dL_dpred  # pred = sum(s_i)
            dw2 += dL_ds * h
            db2[0] += dL_ds
            dL_dh = dL_ds * self.w2
            dL_dpre = dL_dh * (1.0 - h ** 2)
            dW1 += np.outer(dL_dpre, z)
            db1 += dL_dpre
        return {"W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


class KHIPUResonanceNet:
    def __init__(self, d_embed: int, rng: np.random.Generator, temperature: float = 3.0):
        self.bottleneck = State9Bottleneck(d_in=d_embed, rng=rng)
        self.w_resonant = np.array([float(rng.normal(0, 1.0))])
        self.w_other = np.array([float(rng.normal(0, 1.0))])
        self.temperature = temperature

    def params(self):
        p = dict(self.bottleneck.params())
        p["w_resonant"] = self.w_resonant
        p["w_other"] = self.w_other
        return p

    def forward(self, seq: np.ndarray):
        T = seq.shape[0]
        codes = []   # T x (9,) - twarde q (+/-1), to jest FAKTYCZNY wynik uzywany w predykcji
        ts = []      # T x (9,) - ciagle t=tanh(proj) SPRZED kwantyzacji, potrzebne do STE w backward
        for i in range(T):
            q = self.bottleneck.forward(seq[i])
            codes.append(q)
            ts.append(self.bottleneck._cache["t"].copy())

        agree = []
        gate = []
        pred = 0.0
        for i in range(T - 1):
            # UWAGA STE: `agree` w forward liczony jest na TWARDYCH kodach q
            # (tak dziala model naprawde), ale w backward() gradient
            # miedzy dwoma skwantyzowanymi wektorami przechodzi tak, jakby
            # kazdy z nich byl swoim ciaglym `t` (pelny STE przez iloczyn
            # dwoch niezaleznie kwantyzowanych wektorow - standard w
            # binarnych sieciach neuronowych, np. XNOR-Net/BinaryConnect,
            # gdzie gradient przez iloczyn wartosci binarnych liczy sie
            # tak, jakby byly ciagle). Zweryfikowane numerycznym
            # gradient-checkiem w tests/test_gradients_models.py.
            a = float(np.dot(codes[i], codes[i + 1])) / N_AXES   # w [-1,1]
            g = sigmoid(self.temperature * a)
            s = g * self.w_resonant[0] + (1 - g) * self.w_other[0]
            agree.append(a)
            gate.append(g)
            pred += s
        cache = {"codes": codes, "ts": ts, "agree": agree, "gate": gate, "T": T, "seq": seq}
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        """
        WAZNE O STE PRZY ILOCZYNIE DWOCH NIEZALEZNIE KWANTOWANYCH WEKTOROW
        (poprawione po tym, jak pierwsza wersja NIE przechodzila numerycznego
        gradient-checku - patrz tests/test_gradients_models.py i historia
        w git log tego pliku):

        `agree = dot(q_i, q_i+1)/9` jest ILOCZYNEM dwoch wielkosci, z ktorych
        KAZDA jest wynikiem STE (`y = t + stop_gradient(q - t)`, wiec
        FORWARD y == q, ale d(y)/d(t) == 1). Rozniczkujac iloczyn dwoch
        takich obiektow po jednym z nich (np. po y_i), WSPOLCZYNNIKIEM przy
        d(y_i) jest FAKTYCZNA WARTOSC drugiego czynnika w punkcie, w ktorym
        naprawde zostal policzony - czyli q_i+1 (twarde +/-1), NIE t_i+1
        (ciagle). Dopiero PO tym mnozeniu nastepuje STE: d(y_i)/d(t_i) = 1.

        Sprawdzone empirycznie (nie tylko wyprowadzone na papierze):
        wersja z `codes` (twarde q) jako wspolczynnikiem zgadza sie z
        poprawnie zlinearyzowanym numerycznym gradientem STE do ~1e-11.
        Wersja, ktora probowalam podstawic `ts` (ciagle) jako wspolczynnik,
        NIE zgadzala sie (blad ~0.03-0.18) - zostawiona w historii gita
        jako przestroga, ze "intuicyjne" podstawienie STE bywa mylace przy
        iloczynach wielu skwantyzowanych wielkosci.
        """
        T = cache["T"]
        codes = cache["codes"]
        gate = cache["gate"]

        dL_dt = [np.zeros(N_AXES) for _ in range(T)]
        dw_res = np.zeros(1)
        dw_other = np.zeros(1)

        for i in range(T - 1):
            g = gate[i]
            dL_ds = dL_dpred
            dw_res[0] += dL_ds * g
            dw_other[0] += dL_ds * (1 - g)
            dL_dg = dL_ds * (self.w_resonant[0] - self.w_other[0])
            dg_da = self.temperature * g * (1 - g)
            dL_da = dL_dg * dg_da
            # d(agree)/d(y_i) = codes[i+1] (WARTOSC drugiego czynnika, twarda) - patrz docstring
            dL_dt[i] += dL_da * codes[i + 1] / N_AXES
            dL_dt[i + 1] += dL_da * codes[i] / N_AXES

        dWq_total = np.zeros_like(self.bottleneck.Wq)
        dbq_total = np.zeros_like(self.bottleneck.bq)
        for i in range(T):
            self.bottleneck.forward(cache["seq"][i])  # odtworz cache (x, proj, t) dla tokenu i
            _, grads = self.bottleneck.backward(dL_dt[i])
            dWq_total += grads["Wq"]
            dbq_total += grads["bq"]

        return {
            "Wq": dWq_total, "bq": dbq_total,
            "w_resonant": dw_res, "w_other": dw_other,
        }

class KHIPUResonanceNetMLP:
    """
    Wariant KHIPUResonanceNet, ktory zastepuje dwa skalary
    (w_resonant/w_other) pelnym 2-warstwowym MLP nad konkatenacja
    [q_i, q_i+1] (18-wym., dyskretne +/-1) - ten sam rozmiar ukryty co
    BaselinePairwiseMLP, zeby porownanie bylo uczciwe pod wzgledem
    pojemnosci agregacji. Izoluje pytanie: czy KHIPUResonanceNet
    przegrywa z BaselinePairwiseMLP z powodu SLABEJ AGREGACJI (2 skalary),
    czy z powodu SAMEJ KWANTYZACJI (STE)?

    WAZNE uproszczenie wzgledem KHIPUResonanceNet: tutaj NIE MA iloczynu
    dwoch niezaleznie kwantyzowanych wektorow (to byla zrodlo subtelnego
    bledu w gradiencie KHIPUResonanceNet - patrz git log). q_i i q_i+1
    sa tu po prostu SKLEJONE (konkatenacja), potem liniowo zmieszane przez
    W1 - to jest zwykla warstwa liniowa, gradient wzgledem q_i to po prostu
    odpowiedni fragment W1 (bez adnej dodatkowej subtelnosci STE poza
    tym, co juz robi State9Bottleneck.backward). Prostsze i mniej podatne
    na bledy niz wersja z iloczynem skalarnym.
    """

    def __init__(self, d_embed: int, hidden: int, rng: np.random.Generator):
        self.bottleneck = State9Bottleneck(d_in=d_embed, rng=rng)
        d_in = 2 * N_AXES
        scale1 = 1.0 / np.sqrt(d_in)
        scale2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, scale1, size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, size=hidden)
        self.b2 = np.zeros(1)

    def params(self):
        p = dict(self.bottleneck.params())
        p.update({"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2})
        return p

    def forward(self, seq: np.ndarray):
        T = seq.shape[0]
        codes = [self.bottleneck.forward(seq[i]) for i in range(T)]
        cache = {"codes": codes, "seq": seq, "T": T, "z": [], "h": []}
        pred = 0.0
        for i in range(T - 1):
            z = np.concatenate([codes[i], codes[i + 1]])  # (18,)
            h = np.tanh(self.W1 @ z + self.b1)
            s = float(self.w2 @ h + self.b2[0])
            pred += s
            cache["z"].append(z)
            cache["h"].append(h)
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        T = cache["T"]
        codes = cache["codes"]
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros(1)
        dL_dq = [np.zeros(N_AXES) for _ in range(T)]

        for i in range(T - 1):
            z, h = cache["z"][i], cache["h"][i]
            dL_ds = dL_dpred
            dw2 += dL_ds * h
            db2[0] += dL_ds
            dL_dh = dL_ds * self.w2
            dL_dpre = dL_dh * (1.0 - h ** 2)
            dW1 += np.outer(dL_dpre, z)
            db1 += dL_dpre
            dL_dz = self.W1.T @ dL_dpre          # (18,)
            dL_dq[i] += dL_dz[:N_AXES]
            dL_dq[i + 1] += dL_dz[N_AXES:]

        dWq_total = np.zeros_like(self.bottleneck.Wq)
        dbq_total = np.zeros_like(self.bottleneck.bq)
        for i in range(T):
            self.bottleneck.forward(cache["seq"][i])
            _, grads = self.bottleneck.backward(dL_dq[i])
            dWq_total += grads["Wq"]
            dbq_total += grads["bq"]

        return {
            "Wq": dWq_total, "bq": dbq_total,
            "W1": dW1, "b1": db1, "w2": dw2, "b2": db2,
        }

class KHIPUResonanceNetMLPNoQuant:
    """
    ABLACJA: identyczny pipeline co KHIPUResonanceNetMLP, ale BEZ twardej
    kwantyzacji (bez sign() i bez korekty rownowagi F4-RED) - uzywa
    ciaglego `t=tanh(proj)` bezposrednio jako "kodu". Izoluje pytanie:
    czy przewaga KHIPUResonanceNetMLP nad baseline bierze sie z SAMEJ
    DYSKRETYZACJI (hipoteza: odszumianie przez wymuszona decyzje
    kategoryczna), czy tylko z redukcji wymiarowosci do 9 osi (ktora
    sama w sobie tez moze dzialac jak "bottleneck" ograniczajacy przekaz
    szumu, bez dyskretnosci)?

    Jesli ten wariant wypada blisko KHIPUResonanceNetMLP -> przewage
    daje redukcja wymiarowosci, nie dyskretnosc. Jesli wypada blisko
    (albo gorzej niz) baseline -> przewage faktycznie daje dyskretyzacja.
    """

    def __init__(self, d_embed: int, hidden: int, rng: np.random.Generator):
        scale_q = 1.0 / np.sqrt(d_embed)
        self.Wq = rng.normal(0, scale_q, size=(N_AXES, d_embed))
        self.bq = np.zeros(N_AXES)

        d_in = 2 * N_AXES
        scale1 = 1.0 / np.sqrt(d_in)
        scale2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, scale1, size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, size=hidden)
        self.b2 = np.zeros(1)

    def params(self):
        return {"Wq": self.Wq, "bq": self.bq, "W1": self.W1, "b1": self.b1,
                "w2": self.w2, "b2": self.b2}

    def forward(self, seq: np.ndarray):
        T = seq.shape[0]
        ts = []
        for i in range(T):
            proj = self.Wq @ seq[i] + self.bq
            ts.append(np.tanh(proj))  # BEZ sign()/balance_correct - roznica wzgledem MLP-wersji
        cache = {"ts": ts, "seq": seq, "T": T, "z": [], "h": []}
        pred = 0.0
        for i in range(T - 1):
            z = np.concatenate([ts[i], ts[i + 1]])
            h = np.tanh(self.W1 @ z + self.b1)
            s = float(self.w2 @ h + self.b2[0])
            pred += s
            cache["z"].append(z)
            cache["h"].append(h)
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        T = cache["T"]
        ts = cache["ts"]
        seq = cache["seq"]
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros(1)
        dL_dt = [np.zeros(N_AXES) for _ in range(T)]

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
            dL_dt[i] += dL_dz[:N_AXES]
            dL_dt[i + 1] += dL_dz[N_AXES:]

        dWq_total = np.zeros_like(self.Wq)
        dbq_total = np.zeros_like(self.bq)
        for i in range(T):
            t = ts[i]
            dL_dproj = dL_dt[i] * (1.0 - t ** 2)
            dWq_total += np.outer(dL_dproj, seq[i])
            dbq_total += dL_dproj

        return {"Wq": dWq_total, "bq": dbq_total, "W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


class KHIPUResonanceNetStructured:
    """
    Propozycja usprawnienia (2026-08-22), nie ablacja: zmienia KOLEJNOSC
    transformacji i redukcji wzgledem KHIPUResonanceNetMLP.

    KHIPUResonanceNetMLP robi: kwantyzuj kazdy token (transformacja) ->
    SKLEJ dwa kody 9+9=18-wym. (naiwna redukcja bez struktury) -> MLP sam
    musi sie nauczyc z surowej konkatenacji, ze liczy sie ZGODNOSC
    kazdej z 9 osi (funkcja typu XNOR na kazdej osi), a nie ich absolutne
    wartosci czy kolejnosc (i, i+1) vs (i+1, i).

    Tu redukcja jest zrobiona WPROST regula GIPU (per-os "ten sam
    znak -> rezonans"), PRZED MLP, nie zostawiona MLP-owi do odgadniecia:

        r_k = q_i[k] * q_i+1[k]   dla k=1..9   (+1 = zgodne osie, -1 = rozne)

    `r` (9-wym., ta sama informacja co pelna konkatenacja pod wzgledem
    "co jest zgodne", ale w formie ktora juz JEST symetryczna i niezalezna
    od kolejnosci) idzie w mniejszy MLP (9 -> hidden -> 1) zamiast 18 ->
    hidden -> 1 jak w KHIPUResonanceNetMLP - mniej parametrow, bo redukcja
    dokladajaca wiedze o zadaniu (regula GIPU) zamiast czystej pojemnosci.

    STE przez iloczyn q_i[k]*q_i+1[k]: dokladnie ta sama, juz zweryfikowana
    (do ~1e-11 w KHIPUResonanceNet) konwencja - pochodna wzgledem jednego
    czynnika uzywa TWARDEJ wartosci drugiego czynnika jako wspolczynnika,
    potem STE (tozsamosc) az do t=tanh(proj) przed kwantyzacja.
    """

    def __init__(self, d_embed: int, hidden: int, rng: np.random.Generator):
        self.bottleneck = State9Bottleneck(d_in=d_embed, rng=rng)
        d_in = N_AXES
        scale1 = 1.0 / np.sqrt(d_in)
        scale2 = 1.0 / np.sqrt(hidden)
        self.W1 = rng.normal(0, scale1, size=(hidden, d_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, scale2, size=hidden)
        self.b2 = np.zeros(1)

    def params(self):
        p = dict(self.bottleneck.params())
        p.update({"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2})
        return p

    def forward(self, seq: np.ndarray):
        T = seq.shape[0]
        codes = [self.bottleneck.forward(seq[i]) for i in range(T)]
        cache = {"codes": codes, "seq": seq, "T": T, "r": [], "h": []}
        pred = 0.0
        for i in range(T - 1):
            r = codes[i] * codes[i + 1]          # (9,), regula GIPU per-os
            h = np.tanh(self.W1 @ r + self.b1)
            s = float(self.w2 @ h + self.b2[0])
            pred += s
            cache["r"].append(r)
            cache["h"].append(h)
        return pred, cache

    def backward(self, dL_dpred: float, cache):
        T = cache["T"]
        codes = cache["codes"]
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros(1)
        dL_dq = [np.zeros(N_AXES) for _ in range(T)]

        for i in range(T - 1):
            r, h = cache["r"][i], cache["h"][i]
            dL_ds = dL_dpred
            dw2 += dL_ds * h
            db2[0] += dL_ds
            dL_dh = dL_ds * self.w2
            dL_dpre = dL_dh * (1.0 - h ** 2)
            dW1 += np.outer(dL_dpre, r)
            db1 += dL_dpre
            dL_dr = self.W1.T @ dL_dpre          # (9,)
            # d(r_k)/d(q_i[k]) = q_i+1[k] (twarda wartosc drugiego czynnika)
            dL_dq[i] += dL_dr * codes[i + 1]
            dL_dq[i + 1] += dL_dr * codes[i]

        dWq_total = np.zeros_like(self.bottleneck.Wq)
        dbq_total = np.zeros_like(self.bottleneck.bq)
        for i in range(T):
            self.bottleneck.forward(cache["seq"][i])
            _, grads = self.bottleneck.backward(dL_dq[i])
            dWq_total += grads["Wq"]
            dbq_total += grads["bq"]

        return {
            "Wq": dWq_total, "bq": dbq_total,
            "W1": dW1, "b1": db1, "w2": dw2, "b2": db2,
        }
