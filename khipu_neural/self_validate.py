"""
self_validate.py — samowalidacja jako proces samodoskonalenia.

Dwie NIEZALEZNE bramki samosprawdzania, uruchamiane PRZED kazda proba
"samodoskonalenia" (self_improve_bottleneck), plus historia decyzji
zapisywana przyrostowo (append-only) do self_improve_log.json:

1. run_correctness_checks() — male, szybkie gradient-checki (zwykly i
   STE) dla kluczowych klas modeli. Bramka: jesli backward jakiegos
   modelu jest zepsuty (np. przez przyszla zmiane kodu), proces
   samodoskonalenia MA SIE ZATRZYMAC, zamiast "poprawiac" model na
   podstawie zlamanej matematyki. Bez tego "samodoskonalenie" moglo by
   po cichu utrwalic bledny wynik.

2. check_shapes() — sprawdza zgodnosc wymiarow calego pipeline'u
   (Wq@x, W1@z, w2@h) dla danej konfiguracji (d_embed, n_axes, hidden),
   zanim w ogole zaczniemy trening. To lokalny, samodzielny odpowiednik
   tego, co manualnie sprawdzono zewnetrznym math-validatorem
   (jbackk-lang/math-validator-3.0, linalg.validate_matrix_expression,
   "analiza topologiczna" ksztaltow macierzy) — zaimplementowany tutaj
   bez zaleznosci od innego repo, zeby KHIPU-NEURAL zostalo samodzielne.

self_improve_bottleneck() SAMO W SOBIE NIE JEST SLEPE ani chciwe: nie
promuje nowej konfiguracji tylko dlatego, ze wypadla lepiej na garstce
ziaren. Wymaga:
  - obu bramek (correctness + shape) czystych,
  - MINIMALNEJ liczby ziaren (domyslnie 5) — bo w tym repo juz raz
    (ablation.py, bottleneck_sweep.py) widzielismy, ze 3-4 ziarna
    potrafia dac mylacy wynik (khipu_mlp vs noquant byly nierozstrzygniete
    na 4 ziarnach; n_axes=6 mial std=0.10 na samych 3 ziarnach),
  - progu WZGLEDNEJ poprawy (domyslnie 10%) — nie samego "lepszy o cokolwiek".

Kazde uruchomienie dopisuje wpis do self_improve_log.json z decyzja i
uzasadnieniem — to jest "pamiec" procesu w czasie, analogicznie do
logow predykcja/rzeczywistosc w TIMDR (skill: timdr-signal-framework).
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from .bottleneck_sweep import BottleneckMLP, run_one
from .quantize import balance_correct

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "self_improve_log.json")


# ---------------------------------------------------------------------------
# Bramka 1: poprawnosc matematyczna (gradient-checki, szybka wersja)
# ---------------------------------------------------------------------------

def _quantize(Wq, bq, x):
    proj = Wq @ x + bq
    t = np.tanh(proj)
    q = np.sign(t)
    q[q == 0] = 1.0
    q = balance_correct(q, t)
    return t, q


def check_bottleneck_mlp_gradients(n_axes=6, d_embed=3, T=5, hidden=4, seed=99, eps=1e-5, tol=1e-4):
    """Szybki gradient-check BottleneckMLP (STE + zwykly backprop). Zwraca
    dict z max_err per parametr i ogolnym ok=True/False, NIE assertuje —
    to funkcja diagnostyczna do uzycia przez self_improve, nie test pytest
    (ten sam test jest tez w tests/test_bottleneck_sweep_smoke.py jako
    asercja, dla CI)."""
    rng = np.random.default_rng(seed)
    model = BottleneckMLP(d_embed=d_embed, n_axes=n_axes, hidden=hidden, rng=rng)
    seq = rng.normal(size=(T, d_embed))

    pred, cache = model.forward(seq)
    grads = model.backward(1.0, cache)
    errors = {}

    for name in ("W1", "b1", "w2", "b2"):
        param = getattr(model, name)
        flat = param.flatten().copy()
        num_grad = np.zeros_like(flat)
        for i in range(len(flat)):
            plus = flat.copy(); plus[i] += eps
            minus = flat.copy(); minus[i] -= eps

            def set_and_forward(v):
                old = getattr(model, name).copy()
                getattr(model, name)[...] = v.reshape(old.shape)
                p, _ = model.forward(seq)
                getattr(model, name)[...] = old
                return p

            num_grad[i] = (set_and_forward(plus) - set_and_forward(minus)) / (2 * eps)
        errors[name] = float(np.max(np.abs(num_grad - grads[name].flatten())))

    Wq0 = model.Wq.copy()
    bq0 = model.bq.copy()
    t0_list, q0_list = [], []
    for i in range(T):
        t, q = _quantize(Wq0, bq0, seq[i])
        t0_list.append(t); q0_list.append(q)

    def pred_from_codes(codes):
        p = 0.0
        for i in range(T - 1):
            z = np.concatenate([codes[i], codes[i + 1]])
            h = np.tanh(model.W1 @ z + model.b1)
            p += float(model.w2 @ h + model.b2[0])
        return p

    def numerical_grad_wq(idx):
        Wp = Wq0.copy(); Wp[idx] += eps
        Wm = Wq0.copy(); Wm[idx] -= eps
        codes_p, codes_m = [], []
        for i in range(T):
            tp, _ = _quantize(Wp, bq0, seq[i])
            tm, _ = _quantize(Wm, bq0, seq[i])
            codes_p.append(q0_list[i] + (tp - t0_list[i]))
            codes_m.append(q0_list[i] + (tm - t0_list[i]))
        return (pred_from_codes(codes_p) - pred_from_codes(codes_m)) / (2 * eps)

    num_grad_Wq = np.zeros_like(Wq0)
    it = np.nditer(Wq0, flags=["multi_index"])
    for _ in it:
        num_grad_Wq[it.multi_index] = float(numerical_grad_wq(it.multi_index))
    errors["Wq"] = float(np.max(np.abs(num_grad_Wq - grads["Wq"])))

    ok = all(e < tol for e in errors.values())
    return {"ok": ok, "errors": errors, "tol": tol}


def run_correctness_checks(verbose=True):
    """Uruchamia wszystkie dostepne gradient-checki. Zwraca
    {"all_ok": bool, "checks": {...}}. To jest bramka #1 przed
    samodoskonaleniem."""
    checks = {"bottleneck_mlp": check_bottleneck_mlp_gradients()}
    all_ok = all(c["ok"] for c in checks.values())
    if verbose:
        for name, c in checks.items():
            status = "OK" if c["ok"] else "BLAD"
            print(f"  [korektnosc] {name}: {status} (max_err={max(c['errors'].values()):.2e})")
    return {"all_ok": all_ok, "checks": checks}


# ---------------------------------------------------------------------------
# Bramka 2: zgodnosc wymiarow (lokalny odpowiednik math-validatora)
# ---------------------------------------------------------------------------

def check_shapes(d_embed: int, n_axes: int, hidden: int):
    """Sprawdza, ze kazde mnozenie macierzowe w pipeline BottleneckMLP
    (Wq@x, W1@z, w2@h) jest wymiarowo spojne DLA DANEJ KONFIGURACJI,
    zanim zacznie sie trening. Lokalny odpowiednik manualnej walidacji
    zrobionej wczesniej narzedziem jbackk-lang/math-validator-3.0
    (linalg.validate_matrix_expression) - bez zaleznosci od tamtego repo."""
    ok = True
    messages = []
    try:
        Wq = np.zeros((n_axes, d_embed)); x = np.zeros(d_embed)
        assert (Wq @ x).shape == (n_axes,)
    except Exception as e:
        ok = False
        messages.append(f"Wq@x: {e}")
    try:
        W1 = np.zeros((hidden, 2 * n_axes)); z = np.zeros(2 * n_axes)
        assert (W1 @ z).shape == (hidden,)
    except Exception as e:
        ok = False
        messages.append(f"W1@z: {e}")
    try:
        w2 = np.zeros(hidden); h = np.zeros(hidden)
        assert np.ndim(w2 @ h) == 0
    except Exception as e:
        ok = False
        messages.append(f"w2@h: {e}")
    return {"ok": ok, "messages": messages, "config": {"d_embed": d_embed, "n_axes": n_axes, "hidden": hidden}}


# ---------------------------------------------------------------------------
# Proces samodoskonalenia
# ---------------------------------------------------------------------------

def _append_log(log_path, entry):
    entry = dict(entry)
    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    history = []
    if os.path.exists(log_path):
        try:
            history = json.load(open(log_path, encoding="utf-8"))
        except Exception:
            history = []
    history.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def self_improve_bottleneck(candidates=(9, 13), current_default=9, seeds=(1, 2, 3, 4, 5),
                             min_seeds=5, min_relative_improvement=0.10,
                             known_results=None, d_embed=8, seq_len=10, hidden=16,
                             log_path=LOG_PATH, verbose=True):
    """Probuje samodoskonalenia liczby osi bottlenecku. NIE promuje nowej
    konfiguracji, jesli: (a) ktorakolwiek bramka (poprawnosc/wymiary) nie
    przejdzie, (b) mamy mniej niz `min_seeds` ziaren, (c) wzgledna poprawa
    najlepszego kandydata jest ponizej `min_relative_improvement`.

    known_results: opcjonalny slownik {n_axes: {seed: mae}} - juz policzone
    wyniki (np. z bottleneck_sweep.py), zeby nie liczyc ich ponownie.
    Kazde wywolanie dopisuje wpis do self_improve_log.json."""
    all_configs = tuple(sorted(set(candidates) | {current_default}))

    # Bramka 1: poprawnosc
    correctness = run_correctness_checks(verbose=verbose)
    if not correctness["all_ok"]:
        decision = {"decision": "aborted_correctness_failure", "correctness": correctness}
        _append_log(log_path, decision)
        if verbose:
            print("PRZERWANO: gradient-check nie przeszedl - samodoskonalenie zatrzymane.")
        return decision

    # Bramka 2: wymiary, dla kazdej rozwazanej konfiguracji
    for c in all_configs:
        shape_result = check_shapes(d_embed=d_embed, n_axes=c, hidden=hidden)
        if not shape_result["ok"]:
            decision = {"decision": f"aborted_shape_failure_n_axes_{c}", "shape_check": shape_result}
            _append_log(log_path, decision)
            if verbose:
                print(f"PRZERWANO: niezgodnosc wymiarow dla n_axes={c} - {shape_result['messages']}")
            return decision
    if verbose:
        print(f"  [wymiary] wszystkie konfiguracje {all_configs} spojne wymiarowo.")

    # Zbieranie wynikow (uzupelnia braki wzgledem known_results)
    known_results = known_results or {}
    results = {}
    for c in all_configs:
        have = dict(known_results.get(c, {}))
        for s in seeds:
            if s not in have:
                mae, _ = run_one(n_axes=c, seed=s, d_embed=d_embed, seq_len=seq_len, hidden=hidden)
                have[s] = mae
                if verbose:
                    print(f"  [pomiar] n_axes={c} seed={s}: mae={mae:.4f}")
        vals = [have[s] for s in seeds]
        results[c] = {"seeds": list(seeds), "vals": vals,
                      "mean": float(np.mean(vals)), "std": float(np.std(vals))}

    n_seeds = len(seeds)
    if n_seeds < min_seeds:
        decision = {
            "decision": "insufficient_evidence",
            "reason": f"tylko {n_seeds} ziaren (< wymaganych {min_seeds}) - za malo, "
                      f"zeby odroznic prawdziwa poprawe od szumu (patrz ablation.py/bottleneck_sweep.py)",
            "results": results, "current_default": current_default,
        }
        _append_log(log_path, decision)
        if verbose:
            print(decision["reason"])
        return decision

    baseline = results[current_default]
    best_candidate = min(candidates, key=lambda c: results[c]["mean"])
    best = results[best_candidate]
    rel_improve = ((baseline["mean"] - best["mean"]) / baseline["mean"]
                    if baseline["mean"] > 0 else 0.0)

    if best_candidate != current_default and rel_improve >= min_relative_improvement:
        decision = {
            "decision": f"promote:{best_candidate}",
            "from": current_default, "to": best_candidate,
            "relative_improvement": rel_improve, "results": results,
        }
    else:
        decision = {
            "decision": f"keep:{current_default}",
            "reason": f"najlepszy kandydat (n_axes={best_candidate}) poprawia sie o "
                      f"{rel_improve:.1%}, ponizej progu {min_relative_improvement:.0%}",
            "results": results,
        }
    _append_log(log_path, decision)
    if verbose:
        print(decision.get("reason", decision["decision"]))
    return decision


if __name__ == "__main__":
    self_improve_bottleneck()
