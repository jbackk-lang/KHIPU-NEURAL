"""Smoke test for ablation.py: confirms all 4 model variants train and
evaluate without error and produce finite MAE values. Not a statistical
re-verification (that's the point of running ablation.py itself with
more seeds) -- just checks the pipeline doesn't crash and returns
sane numbers.
"""
import math
from khipu_neural.ablation import run


def test_ablation_runs_and_produces_finite_results():
    results = run(seeds=(1,), steps=20, verbose=False)
    assert set(results.keys()) == {
        "baseline_289", "baseline_397", "khipu_mlp", "khipu_noquant"
    }
    for key, vals in results.items():
        assert len(vals) == 1
        assert math.isfinite(vals[0])
        assert vals[0] >= 0.0
