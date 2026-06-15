#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locked v11 one-factor gradient environment runner
=================================================

Purpose
-------
Run the locked v11 autonomous prototype across a preregistered one-factor
3D-environment gradient design. The model is not tuned or edited. The purpose is
not survival maximisation. The purpose is to test whether environment factors
(resource density, danger density, friction/slip, unknown/ambiguity, and vertical
complexity) systematically shift behaviour and internal module dynamics.

Design
------
Full mode uses:
21 environments × 5 tasks × 6 observational arms × 50 seeds × 500 steps
= 15,750,000 requested step updates before early termination.

The 21 environments are baseline plus four non-baseline levels for each of five
one-factor gradients. Full factorial combinations are intentionally avoided.

Computation strategy
--------------------
- Import v11 once.
- Load the DARCA core once.
- Do not repeat Q/physics/social source validation probes per environment.
- Do not include lesion arms by default; this is not a lesion-validation sweep.
- Full step logs are not written by default; episode-level summaries and bounded
  step samples are written.
- All gradient statistics are computed post hoc after the run.

Typical staged runs
-------------------
Unknown-gradient confirmatory run:
python3 -u run_v11_locked_gradient_staged.py \
  --v11-file ~/Downloads/darca_true_3d_integrated_task_battery_v11.py \
  --darca-file ~/Downloads/darca_v24_direct_rewrite_source.py \
  --outdir ~/Desktop/DARCA_V11_UNKNOWN_GRADIENT_50SEED \
  --plan unknown_confirmatory \
  --overwrite

Fast all-factor screening run:
python3 -u run_v11_locked_gradient_staged.py \
  --v11-file ~/Downloads/darca_true_3d_integrated_task_battery_v11.py \
  --darca-file ~/Downloads/darca_v24_direct_rewrite_source.py \
  --outdir ~/Desktop/DARCA_V11_GRADIENT_SCREENING_20SEED \
  --plan screening \
  --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, (float, np.floating)):
                    out[k] = f"{float(v):.10g}" if math.isfinite(float(v)) else ""
                elif isinstance(v, (int, np.integer)):
                    out[k] = int(v)
                else:
                    out[k] = v
            w.writerow(out)


def append_df_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    mode = "a" if path.exists() else "w"
    header = not path.exists()
    df.to_csv(path, index=False, mode=mode, header=header)


def normalized_entropy(vals: Sequence[Any]) -> float:
    s = pd.Series([str(v) for v in vals if str(v) != "nan"])
    if len(s) == 0:
        return 0.0
    counts = s.value_counts().to_numpy(dtype=float)
    p = counts / max(counts.sum(), EPS)
    h = float(-(p * np.log2(p + EPS)).sum())
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h / hmax if hmax > 0 else 0.0


def mean_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).mean())


def sum_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def last_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return safe_float(pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]) if len(df[col]) else 0.0


def bootstrap_ci(x: Sequence[float], seed: int, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray([safe_float(v, np.nan) for v in x], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def describe_vector(x: Sequence[float], seed: int, prefix: str = "") -> Dict[str, float]:
    arr = np.asarray([safe_float(v, np.nan) for v in x], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}n": 0, f"{prefix}mean": 0.0, f"{prefix}sd": 0.0, f"{prefix}sem": 0.0,
            f"{prefix}ci95_low": 0.0, f"{prefix}ci95_high": 0.0,
            f"{prefix}prop_positive": 0.0, f"{prefix}cohen_dz": 0.0,
        }
    lo, hi = bootstrap_ci(arr, seed=seed)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    mean = float(arr.mean())
    return {
        f"{prefix}n": int(len(arr)),
        f"{prefix}mean": mean,
        f"{prefix}sd": sd,
        f"{prefix}sem": sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0,
        f"{prefix}ci95_low": lo,
        f"{prefix}ci95_high": hi,
        f"{prefix}prop_positive": float((arr > 0).mean()),
        f"{prefix}cohen_dz": mean / (sd + EPS),
    }


class Logger:
    def __init__(self, outdir: Path):
        self.outdir = ensure_dir(outdir)
        self.t0 = time.time()
        self.path = outdir / "gradient_runner.log"
        self.path.write_text(
            f"Locked v11 gradient 50-seed runner\nStarted: {now()}\n" + "=" * 80 + "\n",
            encoding="utf-8",
        )

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:9.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class QuietV11Logger:
    def __init__(self, parent: Logger):
        self.parent = parent

    def log(self, msg: str) -> None:
        if msg.startswith("ERROR") or "Interrupted" in msg:
            self.parent.log("[v11] " + msg)


# =============================================================================
# Environment definitions
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    scenario: str
    description: str
    world_size: int = 11
    z_size: int = 7
    danger_frac: float = 0.075
    resource_frac: float = 0.065
    unknown_frac: float = 0.120
    rest_count: int = 4
    false_resource_frac: float = 0.030
    hidden_rest_frac: float = 0.025
    friction_frac: float = 0.060
    crisis_interval: int = 180
    observation_radius: int = 1
    world_seed_offset: int = 0
    gradient_factor: str = "baseline"
    gradient_level: float = 1.0
    gradient_param: str = "none"
    gradient_value: float = 1.0


def _base_params() -> Dict[str, Any]:
    return dict(
        world_size=11,
        z_size=7,
        danger_frac=0.075,
        resource_frac=0.065,
        unknown_frac=0.120,
        rest_count=4,
        false_resource_frac=0.030,
        hidden_rest_frac=0.025,
        friction_frac=0.060,
        crisis_interval=180,
        observation_radius=1,
    )


def built_in_scenarios() -> List[Scenario]:
    """One-factor gradient set: baseline + 5 factors × 4 non-baseline levels."""
    base = _base_params()
    scenarios: List[Scenario] = [Scenario(
        "baseline_3d",
        "Reference TRUE 3D ecology; common baseline for all one-factor gradients.",
        **base,
        world_seed_offset=0,
        gradient_factor="baseline",
        gradient_level=1.0,
        gradient_param="none",
        gradient_value=1.0,
    )]
    levels = [0.50, 0.75, 1.25, 1.50]
    seed_offset = 10000

    def add(name: str, desc: str, factor: str, param: str, level: float, updates: Dict[str, Any]):
        nonlocal seed_offset
        d = dict(base)
        d.update(updates)
        scenarios.append(Scenario(
            name,
            desc,
            **d,
            world_seed_offset=seed_offset,
            gradient_factor=factor,
            gradient_level=float(level),
            gradient_param=param,
            gradient_value=float(updates.get(param, level)) if param in updates else float(level),
        ))
        seed_offset += 10000

    for lv in levels:
        add(f"resource_x{lv:.2f}".replace(".", "p"), f"Resource-density gradient {lv:.2f}× baseline.", "resource_density", "resource_frac", lv, {"resource_frac": max(0.001, base["resource_frac"] * lv)})
    for lv in levels:
        add(f"danger_x{lv:.2f}".replace(".", "p"), f"Danger-density gradient {lv:.2f}× baseline.", "danger_density", "danger_frac", lv, {"danger_frac": min(0.40, max(0.001, base["danger_frac"] * lv))})
    for lv in levels:
        add(f"friction_x{lv:.2f}".replace(".", "p"), f"Friction/slip gradient {lv:.2f}× baseline.", "friction_slip", "friction_frac", lv, {"friction_frac": min(0.40, max(0.001, base["friction_frac"] * lv))})
    for lv in levels:
        add(f"unknown_x{lv:.2f}".replace(".", "p"), f"Unknown/ambiguity gradient {lv:.2f}× baseline.", "unknown_ambiguity", "unknown_frac", lv, {"unknown_frac": min(0.45, max(0.001, base["unknown_frac"] * lv))})
    for lv, z in [(0.50, 4), (0.75, 5), (1.25, 9), (1.50, 11)]:
        add(f"vertical_x{lv:.2f}".replace(".", "p"), f"Vertical-complexity gradient {lv:.2f}× baseline z-size.", "vertical_complexity", "z_size", lv, {"z_size": int(z)})
    return scenarios


def scenario_names_for_factors(factors: Sequence[str]) -> List[str]:
    """Return baseline plus all non-baseline scenarios belonging to selected factors."""
    wanted = ["baseline_3d"]
    for sc in built_in_scenarios():
        if sc.gradient_factor in set(factors) and sc.scenario != "baseline_3d":
            wanted.append(sc.scenario)
    # preserve order while removing duplicates
    out: List[str] = []
    for x in wanted:
        if x not in out:
            out.append(x)
    return out


def profile_defaults(mode: str, plan: str, factors: str = "") -> Tuple[int, int, str, str, List[str]]:
    """Workload defaults.

    plan controls the environment set and default computational size:
    - unknown_confirmatory: baseline + unknown gradient, 50 seeds × 500 steps.
    - screening: all 21 one-factor environments, 20 seeds × 300 steps.
    - selected_factors: baseline + user-selected factors, 50 seeds × 500 steps.
    - full21: all 21 one-factor environments, 50 seeds × 500 steps.
    - smoke: minimal check, 2 seeds × 80 steps.
    """
    tasks = "viability,delayed_memory,exploration_recovery,physics_adaptation,social_reappraisal"
    arms = "DARCA_ONLY,DARCA_Q,DARCA_PHYSICS,DARCA_Q_PHYSICS,DARCA_Q_SOCIAL,DARCA_Q_PHYSICS_SOCIAL"
    all_scenarios = [sc.scenario for sc in built_in_scenarios()]
    unknown_scenarios = scenario_names_for_factors(["unknown_ambiguity"])

    if plan == "smoke" or mode == "smoke":
        return 2, 80, "viability,physics_adaptation", "DARCA_ONLY,DARCA_Q_PHYSICS_SOCIAL", ["baseline_3d", "unknown_x0p50", "unknown_x1p50"]
    if plan == "unknown_confirmatory":
        return 50, 500, tasks, arms, unknown_scenarios
    if plan == "screening":
        # Fast gradient screen; not the final confirmatory run.
        return 20, 300, tasks, arms, all_scenarios
    if plan == "selected_factors":
        fs = [x.strip() for x in factors.split(",") if x.strip()] or ["resource_density", "danger_density", "friction_slip", "vertical_complexity"]
        return 50, 500, tasks, arms, scenario_names_for_factors(fs)
    if plan == "full21":
        return 50, 500, tasks, arms, all_scenarios
    raise ValueError(plan)


# =============================================================================
# Optional fixed-motif support
# =============================================================================



def load_fixed_motifs(path: str, max_motifs: int) -> pd.DataFrame:
    """Optional fixed motif list for secondary analysis.

    Gradient analysis does not require motifs. If no CSV is supplied, return an
    empty motif table with the required columns so no motif selection is performed.
    """
    cols = ["motif_id", "motif_rank", "scenario", "task", "arm", "motif", "exploratory_score"]
    if not path:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(Path(path).expanduser())
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "motif" not in df.columns:
        raise ValueError("Fixed motif CSV must contain a motif column")
    out = df.copy().head(max_motifs)
    # Normalize expected columns.
    if "motif_id" not in out.columns:
        out["motif_id"] = np.arange(1, len(out) + 1)
    if "motif_rank" not in out.columns:
        out["motif_rank"] = out["motif_id"]
    if "scenario" not in out.columns:
        out["scenario"] = ""
    if "task" not in out.columns:
        out["task"] = ""
    if "arm" not in out.columns:
        out["arm"] = ""
    if "exploratory_score" not in out.columns:
        if "candidate_score_no_survival" in out.columns:
            out["exploratory_score"] = out["candidate_score_no_survival"]
        else:
            out["exploratory_score"] = 0.0
    return out[cols].copy()


# =============================================================================
# Locked v11 execution
# =============================================================================

def import_v11(path: Path):
    spec = importlib.util.spec_from_file_location("locked_v11_model_confirmatory", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import v11 file: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["locked_v11_model_confirmatory"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_v11_args(base: argparse.Namespace, scenario: Scenario, seed: int, world_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        outdir=str(base.outdir),
        darca_file=str(base.darca_file),
        tasks=base.tasks,
        arms=base.arms,
        episodes=base.seeds,
        steps=base.steps,
        seed=seed,
        world_seed=world_seed,
        world_size=scenario.world_size,
        z_size=scenario.z_size,
        danger_frac=scenario.danger_frac,
        resource_frac=scenario.resource_frac,
        unknown_frac=scenario.unknown_frac,
        rest_count=scenario.rest_count,
        false_resource_frac=scenario.false_resource_frac,
        hidden_rest_frac=scenario.hidden_rest_frac,
        friction_frac=scenario.friction_frac,
        crisis_interval=scenario.crisis_interval,
        observation_radius=scenario.observation_radius,
        terminal_h=base.terminal_h,
        theta=base.theta,
        causal_horizon=base.causal_horizon,
        recurrent_N=base.recurrent_N,
        block_consult_below_h=0.22,
        max_consecutive_scans=base.max_consecutive_scans,
        progress_every=10**9,
        q_probe_train_steps=0,
        q_probe_n_state=0,
        q_probe_n_history=0,
        q_agency_probe_n=0,
        overwrite=True,
    )


KEEP_STEP_COLS = [
    "scenario", "task", "arm", "seed_id", "episode", "step", "action", "event", "pos_i", "pos_j", "pos_k", "body_h", "terminal",
    "coverage", "resources", "damage", "total_damage", "resource_gain", "recovery_gain", "entered_unknown", "hit_wall", "outcome_friction",
    "Q", "Q_R", "Q_G", "Q_action_possibility", "Q_avoidance_pressure", "Q_agency", "Q_learned_danger", "Q_learned_comfort", "Q_danger_memory", "Q_pain_memory",
    "physics_score", "physics_pred_error", "physics_pred_damage", "physics_pred_gain", "physics_pred_wall",
    "social_signal", "selected_signal_channel", "signal_probability_max", "receiver_heard_count", "receiver_recovery_increment", "receiver_recovery_total",
    "self_appraisal_gap", "relief", "safe_surprise", "q_relief", "safe_context", "danger_context",
    "pressure_danger_pressure", "pressure_resource_pressure", "pressure_unknown_pressure", "pressure_vertical_pressure",
    "darca_autonomy", "darca_causal_engagement", "darca_prediction_error", "darca_memory_force",
]


def normalize_step_df(ts: List[Dict[str, Any]], scenario: Scenario, seed_id: int) -> pd.DataFrame:
    if not ts:
        return pd.DataFrame()
    df = pd.DataFrame(ts)
    df["scenario"] = scenario.scenario
    df["seed_id"] = seed_id
    df["gradient_factor"] = scenario.gradient_factor
    df["gradient_level"] = scenario.gradient_level
    df["gradient_param"] = scenario.gradient_param
    df["gradient_value"] = scenario.gradient_value
    use = [c for c in KEEP_STEP_COLS if c in df.columns] + ["gradient_factor", "gradient_level", "gradient_param", "gradient_value"]
    return df[use].copy()


def motif_count_in_actions(actions: List[str], motif: str) -> Tuple[int, int]:
    parts = motif.split(">")
    L = len(parts)
    n_win = max(0, len(actions) - L + 1)
    if n_win <= 0:
        return 0, 0
    c = 0
    for i in range(n_win):
        if actions[i:i + L] == parts:
            c += 1
    return c, n_win


def episode_behavior_summary(df: pd.DataFrame, scenario: Scenario, task: str, arm: str, seed_id: int) -> Dict[str, Any]:
    actions = df["action"].astype(str) if "action" in df.columns and not df.empty else pd.Series([], dtype=str)
    out = {
        "scenario": scenario.scenario,
        "task": task,
        "arm": arm,
        "seed_id": seed_id,
        "gradient_factor": scenario.gradient_factor,
        "gradient_level": scenario.gradient_level,
        "gradient_param": scenario.gradient_param,
        "gradient_value": scenario.gradient_value,
        "n_steps": int(len(df)),
        "unique_actions": int(actions.nunique()) if len(actions) else 0,
        "action_entropy_norm": normalized_entropy(actions) if len(actions) else 0.0,
        "move_fraction": float(actions.str.startswith("MOVE").mean()) if len(actions) else 0.0,
        "vertical_action_fraction": float(actions.isin(["MOVE_UP", "MOVE_DOWN"]).mean()) if len(actions) else 0.0,
        "scan_fraction": float((actions == "SCAN").mean()) if len(actions) else 0.0,
        "rest_fraction": float((actions == "REST").mean()) if len(actions) else 0.0,
        "dominant_action": actions.value_counts().index[0] if len(actions) else "",
        "dominant_action_fraction": float((actions == actions.value_counts().index[0]).mean()) if len(actions) else 0.0,
        "mean_body_h": mean_col(df, "body_h"),
        "final_body_h": last_col(df, "body_h"),
        "mean_Q": mean_col(df, "Q"),
        "mean_Q_avoidance_pressure": mean_col(df, "Q_avoidance_pressure"),
        "mean_Q_agency": mean_col(df, "Q_agency"),
        "mean_physics_score": mean_col(df, "physics_score"),
        "mean_physics_pred_error": mean_col(df, "physics_pred_error"),
        "social_signal_rate": mean_col(df, "social_signal"),
        "mean_receiver_recovery_increment": mean_col(df, "receiver_recovery_increment"),
        "mean_self_appraisal_gap": mean_col(df, "self_appraisal_gap"),
        "danger_context_fraction": mean_col(df, "danger_context"),
        "safe_context_fraction": mean_col(df, "safe_context"),
        "entered_unknown_fraction": mean_col(df, "entered_unknown"),
        "hit_wall_fraction": mean_col(df, "hit_wall"),
        "mean_darca_autonomy": mean_col(df, "darca_autonomy"),
        "final_coverage": last_col(df, "coverage"),
        "final_total_damage": last_col(df, "total_damage"),
    }
    if all(c in df.columns for c in ["pos_i", "pos_j", "pos_k"]):
        out["unique_positions"] = int(df[["pos_i", "pos_j", "pos_k"]].drop_duplicates().shape[0])
    else:
        out["unique_positions"] = 0
    return out


def fixed_motif_counts_for_episode(df: pd.DataFrame, fixed_motifs: pd.DataFrame, scenario: Scenario, task: str, arm: str, seed_id: int) -> List[Dict[str, Any]]:
    if df.empty or "action" not in df.columns:
        return []
    sub_motifs = fixed_motifs[(fixed_motifs["task"] == task) & (fixed_motifs["arm"] == arm)]
    if sub_motifs.empty:
        return []
    actions = df["action"].astype(str).tolist()
    rows: List[Dict[str, Any]] = []
    for _, m in sub_motifs.iterrows():
        c, nwin = motif_count_in_actions(actions, str(m["motif"]))
        rows.append({
            "motif_id": int(m["motif_id"]),
            "motif_rank": int(m.get("motif_rank", int(m["motif_id"]))),
            "target_scenario": str(m["scenario"]),
            "task": task,
            "arm": arm,
            "motif": str(m["motif"]),
            "scenario": scenario.scenario,
            "seed_id": seed_id,
            "is_target_context": int(scenario.scenario == str(m["scenario"])),
            "count": int(c),
            "windows": int(nwin),
            "frequency": float(c / max(nwin, 1)),
            "episode_mean_Q": mean_col(df, "Q"),
            "episode_mean_physics_score": mean_col(df, "physics_score"),
            "episode_social_signal_rate": mean_col(df, "social_signal"),
            "episode_vertical_action_fraction": float(pd.Series(actions).isin(["MOVE_UP", "MOVE_DOWN"]).mean()) if actions else 0.0,
        })
    return rows


def sample_for_output(df: pd.DataFrame, scenario: str, seed: int, max_rows: int) -> pd.DataFrame:
    if df.empty or max_rows <= 0:
        return pd.DataFrame()
    n = min(len(df), max_rows)
    return df.sample(n=n, random_state=seed) if len(df) > n else df.copy()


def run_locked(v11: Any, darca_module: Any, scenarios: List[Scenario], fixed_motifs: pd.DataFrame, args: argparse.Namespace, logger: Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    episode_rows: List[Dict[str, Any]] = []
    behavior_rows: List[Dict[str, Any]] = []
    motif_rows: List[Dict[str, Any]] = []
    sample_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []

    total = len(scenarios) * len(tasks) * len(arms) * args.seeds
    done = 0
    v11_logger = QuietV11Logger(logger)
    for si, scenario in enumerate(scenarios):
        t0 = time.time()
        logger.log(f"[scenario] {scenario.scenario} seeds={args.seeds} steps={args.steps} tasks={len(tasks)} arms={len(arms)}")
        ns = make_v11_args(args, scenario, args.seed + si * 1000003, args.world_seed + scenario.world_seed_offset)
        sample_budget = args.step_sample_rows_per_scenario
        scenario_sample: List[pd.DataFrame] = []
        for task in tasks:
            for seed_id in range(args.seeds):
                for arm in arms:
                    try:
                        summary, ts, _maps = v11.run_episode(task, arm, seed_id, ns, darca_module, v11_logger)
                        summary = dict(summary)
                        summary.update({"scenario": scenario.scenario, "seed_id": seed_id})
                        episode_rows.append(summary)
                        sdf = normalize_step_df(ts, scenario, seed_id)
                        if not sdf.empty:
                            behavior_rows.append(episode_behavior_summary(sdf, scenario, task, arm, seed_id))
                            motif_rows.extend(fixed_motif_counts_for_episode(sdf, fixed_motifs, scenario, task, arm, seed_id))
                            if args.save_full_step_logs:
                                append_df_csv(args.outdir / "observational_step_timeseries.csv", sdf)
                            if sample_budget > 0:
                                take = min(args.sample_rows_per_run, sample_budget)
                                ss = sample_for_output(sdf, scenario.scenario, args.seed + seed_id, take)
                                scenario_sample.append(ss)
                                sample_budget -= len(ss)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logger.log(f"[error] scenario={scenario.scenario} task={task} seed={seed_id} arm={arm}: {repr(e)}")
                        episode_rows.append({"scenario": scenario.scenario, "task": task, "arm": arm, "seed_id": seed_id, "error": repr(e)})
                    done += 1
                    if done % max(1, args.progress_every_runs) == 0:
                        logger.log(f"[progress] {done}/{total} runs")
        if scenario_sample:
            sample_frames.append(pd.concat(scenario_sample, ignore_index=True))
        elapsed = time.time() - t0
        run_rows.append({"scenario": scenario.scenario, "elapsed_sec": elapsed, "status": "ok", **asdict(scenario)})
        write_csv(args.outdir / "run_index.csv", run_rows)
        pd.DataFrame(episode_rows).to_csv(args.outdir / "observational_episode_summary.csv", index=False)
        pd.DataFrame(behavior_rows).to_csv(args.outdir / "episode_behavior_summary.csv", index=False)
        pd.DataFrame(motif_rows).to_csv(args.outdir / "confirmatory_motif_episode_counts.csv", index=False)
        logger.log(f"[scenario done] {scenario.scenario} elapsed={elapsed:.1f}s")
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return pd.DataFrame(episode_rows), pd.DataFrame(behavior_rows), pd.DataFrame(motif_rows), sample_df


# =============================================================================
# Confirmatory post hoc analysis
# =============================================================================

CONFIRM_METRICS = [
    "action_entropy_norm",
    "vertical_action_fraction",
    "entered_unknown_fraction",
    "mean_Q",
    "mean_Q_avoidance_pressure",
    "mean_physics_score",
    "mean_physics_pred_error",
    "social_signal_rate",
    "mean_receiver_recovery_increment",
    "mean_self_appraisal_gap",
    "unique_positions",
    "mean_darca_autonomy",
]


def summarize_action_repertoire(behavior_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if behavior_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in behavior_df.groupby(["scenario", "task", "arm"], dropna=False):
        sc, task, arm = keys
        row = {"scenario": sc, "task": task, "arm": arm, "n_seeds": int(sub["seed_id"].nunique())}
        for c in CONFIRM_METRICS:
            if c in sub.columns:
                row.update(describe_vector(sub[c].to_numpy(dtype=float), seed=seed, prefix=f"{c}_"))
        rows.append(row)
    return pd.DataFrame(rows)


def environment_shift_reproducibility(behavior_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if behavior_df.empty:
        return pd.DataFrame()
    rows = []
    base = behavior_df[behavior_df["scenario"] == "baseline_3d"].copy()
    key_cols = ["task", "arm", "seed_id"]
    for sc, sub_sc in behavior_df[behavior_df["scenario"] != "baseline_3d"].groupby("scenario", dropna=False):
        merged = sub_sc.merge(base, on=key_cols, suffixes=("", "_baseline"), how="inner")
        if merged.empty:
            continue
        for task in merged["task"].unique():
            for arm in merged["arm"].unique():
                sub = merged[(merged["task"] == task) & (merged["arm"] == arm)]
                if sub.empty:
                    continue
                for c in CONFIRM_METRICS:
                    bc = c + "_baseline"
                    if c not in sub.columns or bc not in sub.columns:
                        continue
                    delta = pd.to_numeric(sub[c], errors="coerce") - pd.to_numeric(sub[bc], errors="coerce")
                    d = {"scenario": sc, "task": task, "arm": arm, "metric": c}
                    d.update(describe_vector(delta.to_numpy(dtype=float), seed=seed, prefix="delta_"))
                    d["mean_value"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
                    d["mean_baseline"] = float(pd.to_numeric(sub[bc], errors="coerce").mean())
                    rows.append(d)
    return pd.DataFrame(rows)


def confirmatory_motif_summary(motif_df: pd.DataFrame, fixed_motifs: pd.DataFrame, seed: int) -> pd.DataFrame:
    if motif_df.empty:
        return pd.DataFrame()
    rows = []
    for _, m in fixed_motifs.iterrows():
        mid = int(m["motif_id"])
        sub = motif_df[motif_df["motif_id"] == mid].copy()
        if sub.empty:
            continue
        target = sub[sub["is_target_context"] == 1].copy()
        background = sub[sub["is_target_context"] == 0].copy()
        # Pair target frequency with the same-seed mean background frequency across non-target scenarios.
        bg_seed = background.groupby("seed_id", dropna=False)["frequency"].mean().reset_index(name="background_frequency") if not background.empty else pd.DataFrame(columns=["seed_id", "background_frequency"])
        tg_seed = target.groupby("seed_id", dropna=False).agg(
            target_frequency=("frequency", "mean"),
            target_count=("count", "sum"),
            target_windows=("windows", "sum"),
            target_Q=("episode_mean_Q", "mean"),
            target_physics_score=("episode_mean_physics_score", "mean"),
            target_social_signal_rate=("episode_social_signal_rate", "mean"),
            target_vertical_fraction=("episode_vertical_action_fraction", "mean"),
        ).reset_index() if not target.empty else pd.DataFrame()
        merged = tg_seed.merge(bg_seed, on="seed_id", how="left").fillna({"background_frequency": 0.0}) if not tg_seed.empty else pd.DataFrame()
        if merged.empty:
            delta = np.array([], dtype=float)
            target_freq = np.array([], dtype=float)
            bg_freq = np.array([], dtype=float)
        else:
            delta = merged["target_frequency"].to_numpy(dtype=float) - merged["background_frequency"].to_numpy(dtype=float)
            target_freq = merged["target_frequency"].to_numpy(dtype=float)
            bg_freq = merged["background_frequency"].to_numpy(dtype=float)
        out = {
            "motif_id": mid,
            "motif_rank": int(m.get("motif_rank", mid)),
            "target_scenario": str(m["scenario"]),
            "task": str(m["task"]),
            "arm": str(m["arm"]),
            "motif": str(m["motif"]),
            "exploratory_score": safe_float(m.get("exploratory_score", 0.0)),
            "n_target_seeds": int(len(target_freq)),
            "target_total_count": int(target["count"].sum()) if not target.empty else 0,
            "target_total_windows": int(target["windows"].sum()) if not target.empty else 0,
            "background_total_count": int(background["count"].sum()) if not background.empty else 0,
            "background_total_windows": int(background["windows"].sum()) if not background.empty else 0,
            "target_hit_rate": float((target.groupby("seed_id")["count"].sum() > 0).mean()) if not target.empty else 0.0,
            "target_frequency_mean": float(np.mean(target_freq)) if len(target_freq) else 0.0,
            "background_frequency_mean": float(np.mean(bg_freq)) if len(bg_freq) else 0.0,
            "log2_enrichment_target_vs_background": math.log2((float(np.mean(target_freq)) + EPS) / (float(np.mean(bg_freq)) + EPS)) if len(target_freq) else 0.0,
            "target_mean_Q": float(merged["target_Q"].mean()) if not merged.empty and "target_Q" in merged else 0.0,
            "target_mean_physics_score": float(merged["target_physics_score"].mean()) if not merged.empty and "target_physics_score" in merged else 0.0,
            "target_social_signal_rate": float(merged["target_social_signal_rate"].mean()) if not merged.empty and "target_social_signal_rate" in merged else 0.0,
            "target_vertical_fraction": float(merged["target_vertical_fraction"].mean()) if not merged.empty and "target_vertical_fraction" in merged else 0.0,
        }
        out.update(describe_vector(delta, seed=seed, prefix="paired_delta_frequency_"))
        out.update(describe_vector(target_freq, seed=seed + 11, prefix="target_frequency_"))
        out.update(describe_vector(bg_freq, seed=seed + 23, prefix="background_frequency_"))
        rows.append(out)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["paired_delta_frequency_mean", "target_total_count"], ascending=[False, False])
    return out_df


def event_action_response_from_sample(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty or not all(c in sample_df.columns for c in ["event", "action"]):
        return pd.DataFrame()
    tmp = sample_df.sort_values([c for c in ["scenario", "task", "arm", "seed_id", "step"] if c in sample_df.columns]).copy()
    gcols = [c for c in ["scenario", "task", "arm", "seed_id"] if c in tmp.columns]
    tmp["next_action"] = tmp.groupby(gcols, dropna=False)["action"].shift(-1) if gcols else tmp["action"].shift(-1)
    rows = []
    for keys, sub in tmp.groupby(["scenario", "task", "arm", "event"], dropna=False):
        if len(sub) < 5:
            continue
        sc, task, arm, event = keys
        na = sub["next_action"].dropna().astype(str)
        dom = na.value_counts().index[0] if len(na) else ""
        rows.append({
            "scenario": sc, "task": task, "arm": arm, "event": event,
            "n_event_steps_sample": int(len(sub)),
            "next_action_entropy_norm": normalized_entropy(na),
            "dominant_next_action": dom,
            "dominant_next_action_fraction": float((na == dom).mean()) if len(na) else 0.0,
            "mean_Q_at_event": mean_col(sub, "Q"),
            "mean_physics_score_at_event": mean_col(sub, "physics_score"),
            "mean_social_signal_at_event": mean_col(sub, "social_signal"),
        })
    return pd.DataFrame(rows)


GRADIENT_METRICS = [
    "action_entropy_norm",
    "vertical_action_fraction",
    "entered_unknown_fraction",
    "mean_Q",
    "mean_Q_avoidance_pressure",
    "mean_Q_agency",
    "mean_physics_score",
    "mean_physics_pred_error",
    "social_signal_rate",
    "mean_receiver_recovery_increment",
    "mean_self_appraisal_gap",
    "danger_context_fraction",
    "safe_context_fraction",
    "unique_positions",
    "mean_darca_autonomy",
]


def _baseline_for_factor(behavior_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    base = behavior_df[behavior_df["scenario"] == "baseline_3d"].copy()
    if base.empty:
        return base
    base["gradient_factor_for_analysis"] = factor
    base["gradient_level_for_analysis"] = 1.0
    return base


def gradient_rows_for_factor(behavior_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    sub = behavior_df[behavior_df["gradient_factor"] == factor].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["gradient_factor_for_analysis"] = factor
    sub["gradient_level_for_analysis"] = pd.to_numeric(sub["gradient_level"], errors="coerce")
    return pd.concat([_baseline_for_factor(behavior_df, factor), sub], ignore_index=True)


def gradient_level_summary(behavior_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if behavior_df.empty or "gradient_factor" not in behavior_df.columns:
        return pd.DataFrame()
    factors = [f for f in sorted(behavior_df["gradient_factor"].dropna().unique()) if f != "baseline"]
    rows: List[Dict[str, Any]] = []
    for factor in factors:
        gdf = gradient_rows_for_factor(behavior_df, factor)
        for keys, sub in gdf.groupby(["gradient_factor_for_analysis", "gradient_level_for_analysis", "scenario", "task", "arm"], dropna=False):
            gf, gl, sc, task, arm = keys
            for metric in GRADIENT_METRICS:
                if metric in sub.columns:
                    d = {"gradient_factor": gf, "gradient_level": float(gl), "scenario": sc, "task": task, "arm": arm, "metric": metric}
                    d.update(describe_vector(pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float), seed=seed, prefix="value_"))
                    rows.append(d)
    return pd.DataFrame(rows)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x) < EPS:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _monotonic_consistency(levels: np.ndarray, values: np.ndarray, slope: float) -> float:
    order = np.argsort(levels)
    vals = values[order]
    dif = np.diff(vals)
    dif = dif[np.isfinite(dif)]
    if len(dif) == 0 or abs(slope) < EPS:
        return 0.0
    return float((np.sign(dif) == np.sign(slope)).mean())


def gradient_slope_summary(behavior_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if behavior_df.empty or "gradient_factor" not in behavior_df.columns:
        return pd.DataFrame()
    factors = [f for f in sorted(behavior_df["gradient_factor"].dropna().unique()) if f != "baseline"]
    rows: List[Dict[str, Any]] = []
    for factor in factors:
        gdf = gradient_rows_for_factor(behavior_df, factor)
        for task in sorted(gdf["task"].dropna().unique()):
            for arm in sorted(gdf["arm"].dropna().unique()):
                sub_ta = gdf[(gdf["task"] == task) & (gdf["arm"] == arm)].copy()
                for metric in GRADIENT_METRICS:
                    if metric not in sub_ta.columns:
                        continue
                    slopes=[]; rs=[]; mons=[]; high_low=[]
                    for sid, sub_seed in sub_ta.groupby("seed_id", dropna=False):
                        tmp = sub_seed[["gradient_level_for_analysis", metric]].copy()
                        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                        tmp["gradient_level_for_analysis"] = pd.to_numeric(tmp["gradient_level_for_analysis"], errors="coerce")
                        tmp = tmp.groupby("gradient_level_for_analysis", as_index=False)[metric].mean().dropna()
                        if len(tmp) < 3:
                            continue
                        x = tmp["gradient_level_for_analysis"].to_numpy(dtype=float)
                        y = tmp[metric].to_numpy(dtype=float)
                        sl = _ols_slope(x, y)
                        slopes.append(sl); rs.append(_pearson_r(x, y)); mons.append(_monotonic_consistency(x, y, sl))
                        low = tmp.loc[tmp["gradient_level_for_analysis"].idxmin(), metric]
                        high = tmp.loc[tmp["gradient_level_for_analysis"].idxmax(), metric]
                        high_low.append(float(high - low))
                    if slopes:
                        row = {"gradient_factor": factor, "task": task, "arm": arm, "metric": metric}
                        row.update(describe_vector(slopes, seed=seed, prefix="slope_"))
                        row.update(describe_vector(rs, seed=seed + 17, prefix="pearson_r_"))
                        row.update(describe_vector(mons, seed=seed + 31, prefix="monotonic_consistency_"))
                        row.update(describe_vector(high_low, seed=seed + 47, prefix="high_minus_low_"))
                        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_slope_mean"] = out["slope_mean"].abs()
        out = out.sort_values(["abs_slope_mean", "slope_prop_positive"], ascending=[False, False])
    return out


def module_to_action_gradient_links(behavior_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if behavior_df.empty or "gradient_factor" not in behavior_df.columns:
        return pd.DataFrame()
    module_metrics = ["mean_Q", "mean_physics_score", "social_signal_rate", "mean_receiver_recovery_increment"]
    action_metrics = ["action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "unique_positions"]
    rows: List[Dict[str, Any]] = []
    factors = [f for f in sorted(behavior_df["gradient_factor"].dropna().unique()) if f != "baseline"]
    for factor in factors:
        gdf = gradient_rows_for_factor(behavior_df, factor)
        for task in sorted(gdf["task"].dropna().unique()):
            for arm in sorted(gdf["arm"].dropna().unique()):
                sub_ta = gdf[(gdf["task"] == task) & (gdf["arm"] == arm)].copy()
                slopes_by_seed: Dict[Any, Dict[str, float]] = {}
                for sid, sub_seed in sub_ta.groupby("seed_id", dropna=False):
                    dd: Dict[str, float] = {}
                    for metric in module_metrics + action_metrics:
                        if metric not in sub_seed.columns:
                            continue
                        tmp = sub_seed[["gradient_level_for_analysis", metric]].copy()
                        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                        tmp["gradient_level_for_analysis"] = pd.to_numeric(tmp["gradient_level_for_analysis"], errors="coerce")
                        tmp = tmp.groupby("gradient_level_for_analysis", as_index=False)[metric].mean().dropna()
                        if len(tmp) >= 3:
                            dd[metric] = _ols_slope(tmp["gradient_level_for_analysis"].to_numpy(dtype=float), tmp[metric].to_numpy(dtype=float))
                    if dd:
                        slopes_by_seed[sid] = dd
                sids = list(slopes_by_seed)
                for mm in module_metrics:
                    for am in action_metrics:
                        x = np.array([slopes_by_seed[s].get(mm, np.nan) for s in sids], dtype=float)
                        y = np.array([slopes_by_seed[s].get(am, np.nan) for s in sids], dtype=float)
                        mask = np.isfinite(x) & np.isfinite(y)
                        rows.append({
                            "gradient_factor": factor, "task": task, "arm": arm,
                            "module_slope_metric": mm, "action_slope_metric": am,
                            "n_seeds": int(mask.sum()), "slope_correlation_r": _pearson_r(x, y),
                            "same_sign_fraction": float((np.sign(x[mask]) == np.sign(y[mask])).mean()) if mask.sum() else 0.0,
                        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("slope_correlation_r", key=lambda s: s.abs(), ascending=False)
    return out


def write_gradient_report(outdir: Path, args: argparse.Namespace, scenarios: List[Scenario], grad_level: pd.DataFrame, grad_slope: pd.DataFrame, link_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("Locked v11 staged one-factor gradient report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append("The v11 autonomous prototype and DARCA core were locked. Only one environment factor was varied at a time. Full factorial combinations were not used.")
    lines.append(f"environments={len(scenarios)}, tasks={args.tasks}, arms={args.arms}, seeds={args.seeds}, steps={args.steps}")
    lines.append(f"requested_step_updates={len(scenarios) * len(args.tasks.split(',')) * len(args.arms.split(',')) * args.seeds * args.steps:,}")
    lines.append("")
    lines.append("Environment gradients")
    lines.append("---------------------")
    man = pd.DataFrame([asdict(s) for s in scenarios])
    lines.append(man[["scenario", "gradient_factor", "gradient_level", "gradient_param", "gradient_value", "description"]].to_string(index=False))
    lines.append("")
    lines.append("Strongest gradient slopes")
    lines.append("--------------------------")
    if grad_slope.empty:
        lines.append("(none)")
    else:
        use = grad_slope[grad_slope["metric"].isin(["action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "mean_Q", "mean_physics_score", "social_signal_rate", "mean_receiver_recovery_increment"])].copy()
        use = use.sort_values("abs_slope_mean", ascending=False).head(35)
        cols = ["gradient_factor", "task", "arm", "metric", "slope_mean", "slope_ci95_low", "slope_ci95_high", "slope_prop_positive", "high_minus_low_mean", "monotonic_consistency_mean"]
        lines.append(use[[c for c in cols if c in use.columns]].to_string(index=False))
    lines.append("")
    lines.append("Module-to-action gradient links")
    lines.append("-------------------------------")
    if link_df.empty:
        lines.append("(none)")
    else:
        cols = ["gradient_factor", "task", "arm", "module_slope_metric", "action_slope_metric", "n_seeds", "slope_correlation_r", "same_sign_fraction"]
        lines.append(link_df[[c for c in cols if c in link_df.columns]].head(35).to_string(index=False))
    lines.append("")
    lines.append("Guardrails")
    lines.append("----------")
    lines.append("- These are environment-response curves of a locked model, not model tuning.")
    lines.append("- Survival, terminal state, and damage are logged variables only, not objectives.")
    lines.append("- Interpret slopes as post hoc environment-conditioned associations unless a preregistered causal intervention is added later.")
    (outdir / "posthoc_gradient_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_report(outdir: Path, args: argparse.Namespace, scenarios: List[Scenario], fixed_motifs: pd.DataFrame, action_summary: pd.DataFrame, shift_df: pd.DataFrame, motif_summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("Locked v11 staged gradient environment-observation report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append("The v11 autonomous prototype and DARCA core were locked. This confirmatory run changed only predefined 3D environment conditions, tasks, arms, seeds, and episode length.")
    lines.append("The objective was post hoc reproducibility of environment-conditioned behaviour and fixed action motifs, not survival maximisation.")
    lines.append("")
    lines.append("Effective workload")
    lines.append("------------------")
    lines.append(f"scenarios={len(scenarios)}, tasks={args.tasks}, arms={args.arms}, seeds={args.seeds}, steps={args.steps}")
    lines.append(f"requested_step_updates={len(scenarios) * len(args.tasks.split(',')) * len(args.arms.split(',')) * args.seeds * args.steps:,}")
    lines.append("")
    lines.append("Computation strategy")
    lines.append("--------------------")
    lines.append("- v11 imported once; DARCA loaded once.")
    lines.append("- Q/physics/social source validation probes were not repeated.")
    lines.append("- One-factor environment gradients were fixed before this run.")
    lines.append("- Full step logs are optional; default output uses compressed episode summaries and bounded samples.")
    lines.append("")
    lines.append("Fixed motifs")
    lines.append("------------")
    lines.append(fixed_motifs[[c for c in ["motif_id", "target_scenario", "scenario", "task", "arm", "motif", "exploratory_score"] if c in fixed_motifs.columns]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Top reproducible fixed motifs")
    lines.append("-----------------------------")
    if motif_summary.empty:
        lines.append("(none)")
    else:
        cols = ["motif_id", "target_scenario", "task", "arm", "motif", "target_hit_rate", "target_frequency_mean", "background_frequency_mean", "paired_delta_frequency_mean", "paired_delta_frequency_ci95_low", "paired_delta_frequency_ci95_high", "paired_delta_frequency_prop_positive", "log2_enrichment_target_vs_background"]
        lines.append(motif_summary[[c for c in cols if c in motif_summary.columns]].head(20).to_string(index=False))
    lines.append("")
    lines.append("Strongest paired environment shifts")
    lines.append("-----------------------------------")
    if shift_df.empty:
        lines.append("(none)")
    else:
        use = shift_df[shift_df["metric"].isin(["action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "mean_Q", "mean_physics_score", "social_signal_rate"])].copy()
        use["abs_delta"] = use["delta_mean"].abs()
        use = use.sort_values(["abs_delta", "delta_prop_positive"], ascending=[False, False]).head(30)
        cols = ["scenario", "task", "arm", "metric", "delta_mean", "delta_ci95_low", "delta_ci95_high", "delta_prop_positive", "delta_cohen_dz"]
        lines.append(use[[c for c in cols if c in use.columns]].to_string(index=False))
    lines.append("")
    lines.append("Guardrails")
    lines.append("----------")
    lines.append("- Do not tune v11 after reading this confirmatory output.")
    lines.append("- Motif reproducibility is behavioural-pattern evidence, not intentionality by itself.")
    lines.append("- Survival, terminal state, and damage are logged variables only; they are not the objective of this confirmatory run.")
    (outdir / "posthoc_confirmatory_report.txt").write_text("\n".join(lines), encoding="utf-8")


def make_figures(outdir: Path, action_summary: pd.DataFrame, shift_df: pd.DataFrame, motif_summary: pd.DataFrame) -> None:
    if plt is None:
        return
    figdir = ensure_dir(outdir / "figures")
    if not shift_df.empty:
        for metric, fname, ylabel in [
            ("action_entropy_norm", "Fig1_confirmatory_action_entropy_delta.png", "paired delta vs baseline"),
            ("vertical_action_fraction", "Fig2_confirmatory_vertical_action_delta.png", "paired delta vs baseline"),
            ("mean_Q", "Fig3_confirmatory_Q_delta.png", "paired delta vs baseline"),
            ("mean_physics_score", "Fig4_confirmatory_physics_delta.png", "paired delta vs baseline"),
            ("social_signal_rate", "Fig5_confirmatory_social_signal_delta.png", "paired delta vs baseline"),
        ]:
            sub = shift_df[shift_df["metric"] == metric].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("delta_mean", ascending=False).head(25)
            labels = [f"{r['scenario']}|{r['task']}|{r['arm']}" for _, r in sub.iterrows()]
            x = np.arange(len(sub))
            plt.figure(figsize=(13, 5))
            plt.bar(x, sub["delta_mean"].values)
            plt.xticks(x, labels, rotation=60, ha="right")
            plt.ylabel(ylabel)
            plt.title(metric)
            plt.tight_layout()
            plt.savefig(figdir / fname, dpi=170)
            plt.close()
    if not motif_summary.empty:
        sub = motif_summary.sort_values("paired_delta_frequency_mean", ascending=False).head(20)
        labels = [f"#{int(r['motif_id'])} {r['target_scenario']}|{r['motif']}" for _, r in sub.iterrows()]
        y = np.arange(len(sub))
        plt.figure(figsize=(12, max(5, 0.35 * len(sub))))
        plt.barh(y, sub["paired_delta_frequency_mean"].values)
        plt.yticks(y, labels)
        plt.xlabel("target frequency minus same-task/arm background")
        plt.title("Confirmatory fixed-motif enrichment")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(figdir / "Fig6_confirmatory_fixed_motif_enrichment.png", dpi=170)
        plt.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locked-v11 staged one-factor gradient runner: unknown-gradient confirmatory + fast all-factor screening.")
    p.add_argument("--v11-file", required=True)
    p.add_argument("--darca-file", required=True)
    p.add_argument("--outdir", default="DARCA_V11_UNKNOWN_GRADIENT_50SEED")
    p.add_argument("--mode", choices=["smoke", "quick", "full"], default="full", help="Compatibility workload flag. Prefer --plan for new runs.")
    p.add_argument("--plan", choices=["unknown_confirmatory", "screening", "selected_factors", "full21", "smoke"], default="unknown_confirmatory", help="Staged gradient plan. Default: unknown_confirmatory.")
    p.add_argument("--factors", default="", help="For --plan selected_factors: comma-separated gradient factors. Options: resource_density,danger_density,friction_slip,unknown_ambiguity,vertical_complexity")
    p.add_argument("--seeds", type=int, default=None, help="Independent seeds per scenario-task-arm. Full default is 50.")
    p.add_argument("--episodes", type=int, default=None, help="Alias for --seeds for compatibility.")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--tasks", default=None)
    p.add_argument("--arms", default=None)
    p.add_argument("--scenario-subset", default="")
    p.add_argument("--fixed-motifs-csv", default="", help="CSV from exploratory emergent_behavior_candidates.csv. If omitted, built-in preselected motifs are used.")
    p.add_argument("--max-fixed-motifs", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--world-seed", type=int, default=9001)
    p.add_argument("--theta", type=float, default=0.70)
    p.add_argument("--causal-horizon", type=int, default=12)
    p.add_argument("--recurrent-N", type=int, default=96)
    p.add_argument("--terminal-h", type=float, default=0.05)
    p.add_argument("--max-consecutive-scans", type=int, default=3)
    p.add_argument("--progress-every-runs", type=int, default=100)
    p.add_argument("--step-sample-rows-per-scenario", type=int, default=10000)
    p.add_argument("--sample-rows-per-run", type=int, default=20)
    p.add_argument("--save-full-step-logs", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir = Path(args.outdir).expanduser()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists and is not empty: {args.outdir}. Use --overwrite.")
        shutil.rmtree(args.outdir)
    ensure_dir(args.outdir)
    logger = Logger(args.outdir)

    v11_file = Path(args.v11_file).expanduser().resolve()
    darca_file = Path(args.darca_file).expanduser().resolve()
    args.darca_file = darca_file
    if not v11_file.exists():
        raise FileNotFoundError(v11_file)
    if not darca_file.exists():
        raise FileNotFoundError(darca_file)

    default_seeds, default_steps, default_tasks, default_arms, subset = profile_defaults(args.mode, args.plan, args.factors)
    if args.episodes is not None and args.seeds is None:
        args.seeds = args.episodes
    args.seeds = int(args.seeds if args.seeds is not None else default_seeds)
    args.steps = int(args.steps if args.steps is not None else default_steps)
    args.tasks = args.tasks or default_tasks
    args.arms = args.arms or default_arms
    scenarios = built_in_scenarios()
    if args.scenario_subset.strip():
        wanted = [x.strip() for x in args.scenario_subset.split(",") if x.strip()]
    else:
        wanted = subset
    scenarios = [s for s in scenarios if s.scenario in set(wanted)]
    missing = sorted(set(wanted) - {s.scenario for s in scenarios})
    if missing:
        raise ValueError(f"Unknown scenarios: {missing}")

    fixed_motifs = load_fixed_motifs(args.fixed_motifs_csv, args.max_fixed_motifs)
    # Normalize target_scenario column in output while preserving input compatibility.
    fixed_motifs["target_scenario"] = fixed_motifs["scenario"]
    fixed_motifs.to_csv(args.outdir / "fixed_confirmatory_motifs.csv", index=False)
    write_csv(args.outdir / "scenario_manifest.csv", [asdict(s) for s in scenarios])

    n_runs = len(scenarios) * len([x for x in args.tasks.split(',') if x.strip()]) * len([x for x in args.arms.split(',') if x.strip()]) * args.seeds
    n_steps = n_runs * args.steps
    lock = {
        "created_at": now(),
        "purpose": "locked-v11 staged one-factor gradient environment-conditioned behaviour observation",
        "plan": args.plan,
        "factors": args.factors,
        "v11_file": str(v11_file),
        "v11_sha256": sha256_file(v11_file),
        "darca_file": str(darca_file),
        "darca_sha256": sha256_file(darca_file),
        "mode": args.mode,
        "seeds": args.seeds,
        "steps": args.steps,
        "tasks": args.tasks,
        "arms": args.arms,
        "requested_step_updates": n_steps,
        "fixed_motifs_source": args.fixed_motifs_csv or "built_in_from_prior_exploratory_run",
        "compute_strategy": [
            "import v11 once",
            "load DARCA once",
            "do not repeat source validation probes",
            "use preregistered one-factor environment gradients",
            "optionally evaluate fixed motifs without selecting new motifs",
            "aggregate post hoc analysis once",
            "save bounded step sample by default",
        ],
    }
    (args.outdir / "model_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.log(f"locked v11={lock['v11_sha256'][:16]}... darca={lock['darca_sha256'][:16]}...")
    logger.log(f"plan={args.plan} mode={args.mode} scenarios={len(scenarios)} seeds={args.seeds} steps={args.steps} runs={n_runs} requested_step_updates={n_steps:,}")
    logger.log(f"fixed motifs for optional secondary analysis={len(fixed_motifs)}")

    v11 = import_v11(v11_file)
    darca_module = v11.load_darca_module(str(darca_file))
    episode_df, behavior_df, motif_df, sample_df = run_locked(v11, darca_module, scenarios, fixed_motifs, args, logger)

    episode_df.to_csv(args.outdir / "observational_episode_summary.csv", index=False)
    behavior_df.to_csv(args.outdir / "episode_behavior_summary.csv", index=False)
    motif_df.to_csv(args.outdir / "confirmatory_motif_episode_counts.csv", index=False)
    if not sample_df.empty:
        sample_df.to_csv(args.outdir / "observational_step_sample.csv", index=False)

    logger.log("[posthoc] summarize confirmatory outputs")
    action_summary = summarize_action_repertoire(behavior_df, seed=args.seed)
    action_summary.to_csv(args.outdir / "action_repertoire_confirmatory_summary.csv", index=False)
    shift_df = environment_shift_reproducibility(behavior_df, seed=args.seed)
    shift_df.to_csv(args.outdir / "environment_shift_reproducibility.csv", index=False)
    grad_level = gradient_level_summary(behavior_df, seed=args.seed)
    grad_level.to_csv(args.outdir / "gradient_level_summary.csv", index=False)
    grad_slope = gradient_slope_summary(behavior_df, seed=args.seed)
    grad_slope.to_csv(args.outdir / "gradient_slope_summary.csv", index=False)
    link_df = module_to_action_gradient_links(behavior_df, seed=args.seed)
    link_df.to_csv(args.outdir / "module_to_action_gradient_links.csv", index=False)
    motif_summary = confirmatory_motif_summary(motif_df, fixed_motifs, seed=args.seed)
    motif_summary.to_csv(args.outdir / "confirmatory_motif_summary.csv", index=False)
    event_df = event_action_response_from_sample(sample_df)
    event_df.to_csv(args.outdir / "event_action_response_sample_summary.csv", index=False)
    write_report(args.outdir, args, scenarios, fixed_motifs, action_summary, shift_df, motif_summary)
    write_gradient_report(args.outdir, args, scenarios, grad_level, grad_slope, link_df)
    make_figures(args.outdir, action_summary, shift_df, motif_summary)
    logger.log(f"[complete] outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
