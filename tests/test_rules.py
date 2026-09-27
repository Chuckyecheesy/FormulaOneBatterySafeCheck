"""Stage 1 and Stage 2 rule tests (spec/06-test-cases.md, S1-* and S2-*)."""

import pytest

from formulatech.config import load_thresholds
from formulatech.rules import evaluate_stages_1_2, thermal_quantities


@pytest.fixture(scope="module")
def cfg():
    return load_thresholds()


@pytest.mark.parametrize(
    "case, V, I, r1_fails",
    [
        ("S1-1", 5.0, -1.2, True),
        ("S1-2", 4.21, -0.01, True),
        ("S1-3", 4.2, -1.0, False),  # V exactly at threshold, strict >
        ("S1-4", 5.0, 0.0, False),  # I exactly at threshold, strict <
        ("S1-5", 5.0, 0.5, False),
        ("S1-6", 3.8, -2.0, False),
    ],
)
def test_stage1(cfg, case, V, I, r1_fails):
    out = evaluate_stages_1_2(V, I, 25.0, 60, cfg)
    r1 = out.checks[0]
    assert r1.code == "OVERCHARGED"
    assert r1.passed is not r1_fails
    if r1_fails:
        # Stage 2 must not run
        assert out.stopped_at_stage == 1
        assert out.checks == [r1]
        assert r1.recorded == {"V": V, "I": I}
        assert r1.thresholds == {"V": "> 4.2", "I": "< 0"}
    else:
        assert len(out.checks) == 4


@pytest.mark.parametrize(
    "case, T, minutes, failed_codes",
    [
        ("S2-1", 25.0, 60, set()),
        ("S2-2", 25.3, 60, set()),  # T_chem exactly 0.3 after rounding
        ("S2-3", 25.31, 60, {"HIGH_TCHEM"}),
        ("S2-4", 30.0, 1, {"HIGH_DTDT", "HIGH_D2TDT2", "HIGH_TCHEM"}),
        ("S2-5", 20.0, 60, set()),  # cooler than T0
        ("S2-6", 25.2, 0.1, {"HIGH_DTDT", "HIGH_D2TDT2"}),
        # A single thermal failure is enough to stop (issue #3): R3 alone, t = 15 s
        ("S2-7", 25.3, 0.25, {"HIGH_D2TDT2"}),
    ],
)
def test_stage2(cfg, case, T, minutes, failed_codes):
    out = evaluate_stages_1_2(3.9, 0.5, T, minutes, cfg)
    assert {c.code for c in out.failures} == failed_codes
    assert out.passed is (not failed_codes)
    assert out.stopped_at_stage == (2 if failed_codes else None)


def test_worked_example(cfg):
    # spec/02-safety-rules.md §2.3: T = 40 °C, 5 min
    q = thermal_quantities(40.0, 5, cfg)
    assert q["t_s"] == 300
    assert q["dT_dt"] == pytest.approx(0.05)
    assert q["d2T_dt2"] == pytest.approx(0.000166667)
    assert q["T_chem"] == pytest.approx(15.0)
    out = evaluate_stages_1_2(3.9, 0.5, 40.0, 5, cfg)
    assert [c.code for c in out.failures] == ["HIGH_DTDT", "HIGH_TCHEM"]


def test_thresholds_come_from_config(cfg):
    # Raising T_chem limit makes S2-3 pass: proves no hard-coded threshold.
    relaxed = {**cfg, "thermal": {**cfg["thermal"], "t_chem_max_c": 1.0}}
    assert evaluate_stages_1_2(3.9, 0.5, 25.31, 60, relaxed).passed


@pytest.mark.parametrize("minutes", [0, -5])
def test_non_positive_duration_rejected(cfg, minutes):
    with pytest.raises(ValueError):
        evaluate_stages_1_2(3.9, 0.5, 25.0, minutes, cfg)
