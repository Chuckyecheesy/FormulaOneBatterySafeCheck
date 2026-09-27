"""Stage 1 (overcharge) and Stage 2 (thermal) safety checks (spec/02-safety-rules.md).

These answer "is the battery a fire hazard right now?". Every threshold is read
from spec/thresholds.yaml; none are hard-coded here (CLAUDE.md, FR-6).
"""

from dataclasses import dataclass, field

from formulatech.config import load_thresholds

# Calculated values are rounded before threshold comparison (CLAUDE.md).
COMPARE_DECIMALS = 9
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class CheckResult:
    """The shared check result record (spec/02-safety-rules.md, "Check result record")."""

    rule: str
    code: str
    passed: bool
    recorded: dict[str, float]
    recorded_kind: str  # "input" | "calculated"
    thresholds: dict[str, str]  # name -> "<op> <value>"
    unit: dict[str, str]  # name -> unit, keyed like `recorded`
    reason: str


@dataclass(frozen=True)
class StageOutcome:
    """Result of running Stages 1-2. `passed` means stage 3 may run; it is not a CAN PROCEED verdict."""

    passed: bool
    stopped_at_stage: int | None  # 1 or 2 when a stage failed, else None
    checks: list[CheckResult] = field(default_factory=list)  # every check that ran, pass or fail

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _fmt(value: float) -> str:
    return f"{value:g}"


def check_overcharge(voltage_v: float, current_a: float, cfg: dict) -> CheckResult:
    """R1 OVERCHARGED: fails only when I < current_max_a AND V > voltage_max_v."""
    oc = cfg["overcharge"]
    overcharged = current_a < oc["current_max_a"] and voltage_v > oc["voltage_max_v"]
    return CheckResult(
        rule="R1",
        code="OVERCHARGED",
        passed=not overcharged,
        recorded={"V": voltage_v, "I": current_a},
        recorded_kind="input",
        thresholds={"V": f"> {_fmt(oc['voltage_max_v'])}", "I": f"< {_fmt(oc['current_max_a'])}"},
        unit={"V": "V", "I": "A"},
        reason="Battery is already overcharged",
    )


def thermal_quantities(temperature_c: float, duration_min: float, cfg: dict) -> dict[str, float]:
    """Derived Stage 2 quantities, rounded to COMPARE_DECIMALS (spec/02-safety-rules.md §2.1)."""
    if duration_min <= 0:
        raise ValueError("Charging duration must be greater than 0 minutes.")
    t = duration_min * SECONDS_PER_MINUTE
    dT = temperature_c - cfg["thermal"]["initial_temp_c"]
    return {
        "t_s": t,
        "dT_dt": round(dT / t, COMPARE_DECIMALS),
        "d2T_dt2": round(dT / t / t, COMPARE_DECIMALS),  # not dT / t**2: t**2 underflows to 0 for tiny t
        "T_chem": round(max(0.0, dT), COMPARE_DECIMALS),
    }


def check_thermal(temperature_c: float, duration_min: float, cfg: dict) -> list[CheckResult]:
    """R2-R4. All three are always evaluated so every failure is reported together (FR-2)."""
    th = cfg["thermal"]
    q = thermal_quantities(temperature_c, duration_min, cfg)
    specs = [
        ("R2", "HIGH_DTDT", "dT_dt", th["dT_dt_max_c_per_s"], "°C/s",
         "Thermal stress from battery charging is at high risk"),
        ("R3", "HIGH_D2TDT2", "d2T_dt2", th["d2T_dt2_max_c_per_s2"], "°C/s²",
         "Heat is increasing at a very fast rate while the battery charges"),
        ("R4", "HIGH_TCHEM", "T_chem", th["t_chem_max_c"], "°C",
         "Heat exposure to the surrounding environment is too high"),
    ]
    return [
        CheckResult(
            rule=rule,
            code=code,
            passed=not q[name] > limit,
            recorded={name: q[name]},
            recorded_kind="calculated",
            thresholds={name: f"> {_fmt(limit)}"},
            unit={name: unit},
            reason=reason,
        )
        for rule, code, name, limit, unit, reason in specs
    ]


def evaluate_stages_1_2(
    voltage_v: float,
    current_a: float,
    temperature_c: float,
    duration_min: float,
    cfg: dict | None = None,
) -> StageOutcome:
    """Run Stage 1, then Stage 2 only if Stage 1 passed (FR-1, FR-2).

    Inputs are assumed already validated (spec/01-requirements.md §1.1).
    """
    cfg = cfg if cfg is not None else load_thresholds()

    r1 = check_overcharge(voltage_v, current_a, cfg)
    if not r1.passed:
        return StageOutcome(passed=False, stopped_at_stage=1, checks=[r1])

    thermal = check_thermal(temperature_c, duration_min, cfg)
    checks = [r1, *thermal]
    if not all(c.passed for c in thermal):
        return StageOutcome(passed=False, stopped_at_stage=2, checks=checks)

    return StageOutcome(passed=True, stopped_at_stage=None, checks=checks)
