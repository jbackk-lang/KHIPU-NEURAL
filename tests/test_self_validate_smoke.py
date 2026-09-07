"""Testy dymne dla self_validate.py: obie bramki + proces decyzyjny
(na malych, szybkich parametrach - nie odtwarza pelnego sweepu)."""
import os
import tempfile
from khipu_neural.self_validate import (
    run_correctness_checks, check_shapes, self_improve_bottleneck,
)


def test_correctness_checks_pass():
    result = run_correctness_checks(verbose=False)
    assert result["all_ok"] is True


def test_shape_check_ok_for_valid_config():
    result = check_shapes(d_embed=8, n_axes=9, hidden=16)
    assert result["ok"] is True


def test_self_improve_reports_insufficient_evidence_with_few_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "log.json")
        decision = self_improve_bottleneck(
            candidates=(9,), current_default=9, seeds=(1, 2),
            min_seeds=5, log_path=log_path, verbose=False,
        )
        assert decision["decision"] == "insufficient_evidence"
        assert os.path.exists(log_path)


def test_self_improve_log_is_append_only():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "log.json")
        self_improve_bottleneck(candidates=(9,), current_default=9, seeds=(1,),
                                 min_seeds=5, log_path=log_path, verbose=False)
        self_improve_bottleneck(candidates=(9,), current_default=9, seeds=(1,),
                                 min_seeds=5, log_path=log_path, verbose=False)
        import json
        history = json.load(open(log_path))
        assert len(history) == 2
