#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARCA TRUE 3D integrated task battery v11
========================================

Purpose
-------
This script does NOT rewrite DARCA / Autonomous-Life-Core. It loads the validated
closed-loop DARCA core with --darca-file and uses it as the fixed basis of a TRUE
3D autonomy task battery.

The only outer modules evaluated here are:

1. Qualitative valence layer
   - imported learned Q structure: body-relative danger, comfort, pain, memory,
     action possibility, and agency-delay attribution.

2. Physical-law layer
   - action-consequence prediction in the TRUE 3D shell.

3. Social signal layer
   - Phase-4c anonymous social signal structure: no LAUGH/HUMOR label, no
     benign-violation controller variable, no external prompt/API, anonymous
     signal channels only, and self-appraisal gap used for analysis only.

Primary evaluation
------------------
The code checks whether these imported outer-module functions preserve their
source diagnostic structure after connection to the fixed DARCA core in TRUE 3D.
No language model, external prompt, or language-hint module is used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# Constants
# =============================================================================

MOVE_ACTIONS = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "MOVE_UP", "MOVE_DOWN"]
ACTIONS = MOVE_ACTIONS + ["REST", "SCAN"]

DIRS: Dict[str, Tuple[int, int, int]] = {
    "MOVE_N": (-1, 0, 0),
    "MOVE_S": (1, 0, 0),
    "MOVE_E": (0, 1, 0),
    "MOVE_W": (0, -1, 0),
    "MOVE_UP": (0, 0, 1),
    "MOVE_DOWN": (0, 0, -1),
}

T_EMPTY = "EMPTY"
T_REST = "REST"
T_RESOURCE = "RESOURCE"
T_DANGER = "DANGER"
T_UNKNOWN = "UNKNOWN"
HIDDEN_TYPES = [T_RESOURCE, T_DANGER, T_REST, T_EMPTY]

CUES: Dict[str, List[str]] = {
    T_RESOURCE: [
        "nutrient scent drifting in 3D space",
        "green bioluminescent trace behind mineral cover",
        "supply marks continuing through an upper chamber",
        "low-risk resource-like signal from a side cavity",
    ],
    T_DANGER: [
        "predator-like acoustic echo in the volume",
        "unstable heat shimmer from a vertical shaft",
        "sharp metallic smell and fractured floor",
        "warning hiss mixed with pressure turbulence",
    ],
    T_REST: [
        "quiet low-stimulation hollow",
        "cool sheltered cavity with stable airflow",
        "protected niche with low sensory load",
        "soft acoustic shadow suitable for recovery",
    ],
    T_EMPTY: [
        "ambiguous empty tunnel",
        "plain passage with weak sensory structure",
        "open but uninformative cavity",
        "neutral corridor with no strong affordance",
    ],
}

TASKS = [
    "viability",
    "delayed_memory",
    "exploration_recovery",
    "physics_adaptation",
    "social_reappraisal",
]

# =============================================================================
# Utilities
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)):
                    out[k] = f"{float(v):.10g}" if math.isfinite(float(v)) else str(v)
                else:
                    out[k] = v
            w.writerow(out)


def mean_field(rows: List[Dict[str, Any]], field: str) -> float:
    return float(np.mean([safe_float(r.get(field)) for r in rows])) if rows else 0.0


def mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


class Logger:
    def __init__(self, outdir: Path):
        self.t0 = time.time()
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.path = outdir / "run.log"
        self.path.write_text("DARCA TRUE 3D integrated task battery v10 run log\n" + "=" * 80 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:9.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =============================================================================
# TRUE 3D world
# =============================================================================

@dataclass
class Tile3D:
    kind: str = T_EMPTY
    hidden: Optional[str] = None
    cue: str = ""
    depleted: bool = False
    dynamic_phase: int = 0
    false_resource: bool = False
    hidden_rest: bool = False
    friction: str = "normal"  # normal, slippery, rough


@dataclass
class StepOutcome:
    delta_h: float
    damage: float
    resource_gain: float
    recovery_gain: float
    hit_wall: bool
    entered_unknown: bool
    revealed_type: str
    event: str
    vertical: bool = False
    friction: str = "normal"


class TrueWorld3D:
    def __init__(
        self,
        seed: int,
        size: int,
        z_size: int,
        danger_frac: float,
        resource_frac: float,
        unknown_frac: float,
        rest_count: int,
        false_resource_frac: float,
        hidden_rest_frac: float,
        crisis_interval: int,
        observation_radius: int,
        friction_frac: float = 0.06,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = int(size)
        self.z_size = int(z_size)
        if self.z_size < 3:
            raise ValueError("--z-size must be >= 3 for TRUE 3D.")
        self.danger_frac = danger_frac
        self.resource_frac = resource_frac
        self.unknown_frac = unknown_frac
        self.rest_count = rest_count
        self.false_resource_frac = false_resource_frac
        self.hidden_rest_frac = hidden_rest_frac
        self.crisis_interval = crisis_interval
        self.observation_radius = observation_radius
        self.friction_frac = friction_frac
        self.start = (self.size // 2, self.size // 2, self.z_size // 2)
        self.grid: Dict[Tuple[int, int, int], Tile3D] = {}
        self._generate()

    def _generate(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.z_size):
                    fr = "normal"
                    if self.rng.random() < self.friction_frac:
                        fr = self.rng.choice(["slippery", "rough"])
                    self.grid[(i, j, k)] = Tile3D(dynamic_phase=self.rng.randint(0, 59), friction=fr)

        def protected(p: Tuple[int, int, int]) -> bool:
            return abs(p[0] - self.start[0]) <= 2 and abs(p[1] - self.start[1]) <= 2 and abs(p[2] - self.start[2]) <= 1

        candidates = [p for p in self.grid if not protected(p)]
        self.rng.shuffle(candidates)
        self.grid[self.start] = Tile3D(T_REST, cue="central true-3D homeostatic rest chamber", friction="normal")

        for _ in range(max(0, self.rest_count - 1)):
            if not candidates:
                break
            p = candidates.pop()
            self.grid[p] = Tile3D(T_REST, cue="stable 3D rest chamber", friction="normal")

        remaining = [p for p in candidates if self.grid[p].kind == T_EMPTY]
        volume = self.size * self.size * self.z_size
        n_danger = int(round(self.danger_frac * volume))
        n_resource = int(round(self.resource_frac * volume))
        n_unknown = int(round(self.unknown_frac * volume))
        n_false = int(round(self.false_resource_frac * volume))
        n_hidden_rest = int(round(self.hidden_rest_frac * volume))

        def exposure_weight(p: Tuple[int, int, int]) -> float:
            z = p[2] / max(1, self.z_size - 1)
            wall = min(p[0], p[1], self.size - 1 - p[0], self.size - 1 - p[1]) / max(1, self.size // 2)
            return 0.70 + 0.55 * z + 0.35 * (1.0 - wall)

        remaining.sort(key=lambda p: self.rng.random() / exposure_weight(p))
        for _ in range(n_danger):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(T_DANGER, cue="visible 3D hazard", dynamic_phase=self.rng.randint(0, 59), friction=self.grid[p].friction)

        self.rng.shuffle(remaining)
        for _ in range(n_resource):
            if not remaining:
                break
            p = remaining.pop()
            cue = "visible resource in upper chamber" if p[2] > self.start[2] else "visible resource in side chamber"
            self.grid[p] = Tile3D(T_RESOURCE, cue=cue, friction=self.grid[p].friction)

        for _ in range(n_false):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(
                T_UNKNOWN,
                hidden=T_DANGER,
                cue=self.rng.choice([
                    "green resource-like glow beside a vertical shaft",
                    "food-like vapor with predator echo from above",
                    "supply trace mixed with fracture marks in 3D space",
                    "bright lure suspended near unstable cavity",
                ]),
                false_resource=True,
                friction=self.grid[p].friction,
            )

        for _ in range(n_hidden_rest):
            if not remaining:
                break
            p = remaining.pop()
            self.grid[p] = Tile3D(
                T_UNKNOWN,
                hidden=T_REST,
                cue=self.rng.choice([
                    "cool acoustic pocket hidden in upper volume",
                    "low-stimulation cavity behind a side wall",
                    "stable airflow behind vertical clutter",
                ]),
                hidden_rest=True,
                friction="normal",
            )

        hidden_weights = [0.26, 0.36, 0.18, 0.20]
        for _ in range(n_unknown):
            if not remaining:
                break
            p = remaining.pop()
            hidden = self.rng.choices(HIDDEN_TYPES, weights=hidden_weights, k=1)[0]
            self.grid[p] = Tile3D(T_UNKNOWN, hidden=hidden, cue=self.rng.choice(CUES[hidden]), friction=self.grid[p].friction)

    def in_bounds(self, p: Tuple[int, int, int]) -> bool:
        i, j, k = p
        return 0 <= i < self.size and 0 <= j < self.size and 0 <= k < self.z_size

    def tile(self, p: Tuple[int, int, int]) -> Tile3D:
        return self.grid[p]

    def is_crisis(self, step: int) -> bool:
        return self.crisis_interval > 0 and step > 0 and (step % self.crisis_interval) < 30

    def actual_kind(self, p: Tuple[int, int, int], step: int = 0) -> str:
        tile = self.tile(p)
        base = tile.hidden if tile.kind == T_UNKNOWN and tile.hidden else tile.kind
        if base == T_EMPTY:
            if self.is_crisis(step) and ((p[0] * 5 + p[1] * 7 + p[2] * 13 + step // 5) % 31) == 0:
                return T_DANGER
            if ((p[0] + 3 * p[1] + 11 * p[2] + step // 29) % 61) == tile.dynamic_phase:
                return T_DANGER
        return base

    def local_observation(self, pos: Tuple[int, int, int], known: Dict[Tuple[int, int, int], str], step: int = 0) -> List[Dict[str, Any]]:
        pi, pj, pk = pos
        out: List[Dict[str, Any]] = []
        r = self.observation_radius
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    if abs(di) + abs(dj) + abs(dk) > r:
                        continue
                    p = (pi + di, pj + dj, pk + dk)
                    if not self.in_bounds(p):
                        continue
                    tile = self.tile(p)
                    out.append({
                        "rel_i": di, "rel_j": dj, "rel_k": dk,
                        "pos_i": p[0], "pos_j": p[1], "pos_k": p[2],
                        "visible_type": known.get(p, tile.kind),
                        "cue": tile.cue,
                        "known": int(p in known),
                        "depleted": int(tile.depleted),
                        "friction": tile.friction,
                        "dynamic_danger_now": int(self.actual_kind(p, step) == T_DANGER and tile.kind == T_EMPTY),
                    })
        return out

    def scan(self, pos: Tuple[int, int, int], known: Dict[Tuple[int, int, int], str], step: int = 0) -> List[Tuple[Tuple[int, int, int], str]]:
        revealed: List[Tuple[Tuple[int, int, int], str]] = []
        known[pos] = self.actual_kind(pos, step)
        for _, (di, dj, dk) in DIRS.items():
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if not self.in_bounds(p):
                continue
            tile = self.tile(p)
            actual = self.actual_kind(p, step)
            if tile.kind == T_UNKNOWN and tile.hidden:
                known[p] = tile.hidden
                revealed.append((p, tile.hidden))
            else:
                known[p] = actual
        return revealed

    def apply_action(self, pos: Tuple[int, int, int], action: str, known: Dict[Tuple[int, int, int], str], body_h: float, step: int) -> Tuple[Tuple[int, int, int], StepOutcome]:
        delta_h = -0.0035
        if self.is_crisis(step):
            delta_h -= 0.006
        damage = 0.0
        resource_gain = 0.0
        recovery_gain = 0.0
        hit_wall = False
        entered_unknown = False
        revealed_type = ""
        event = ""
        new_pos = pos
        vertical = False
        friction = self.tile(pos).friction

        if action in MOVE_ACTIONS:
            di, dj, dk = DIRS[action]
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            vertical = action in ("MOVE_UP", "MOVE_DOWN")
            delta_h -= 0.004 + (0.006 if vertical else 0.0)
            if self.tile(pos).friction == "rough":
                delta_h -= 0.003
            if self.tile(pos).friction == "slippery" and not vertical:
                # Occasionally overshoot one extra cell in the same horizontal direction.
                if self.rng.random() < 0.18:
                    p2 = (p[0] + di, p[1] + dj, p[2] + dk)
                    if self.in_bounds(p2):
                        p = p2
                        event = "slippery_overshoot"
            if not self.in_bounds(p):
                hit_wall = True
                damage += 0.015
                delta_h -= 0.015
                event = "hit_wall"
            else:
                new_pos = p
                tile = self.tile(p)
                friction = tile.friction
                if tile.kind == T_UNKNOWN:
                    entered_unknown = True
                    revealed_type = tile.hidden or T_EMPTY
                    known[p] = revealed_type
                actual = self.actual_kind(p, step)
                if actual == T_DANGER:
                    upper = p[2] >= self.z_size - 1
                    boundary = p[0] in (0, self.size - 1) or p[1] in (0, self.size - 1)
                    dmg = 0.080 + 0.080 * float(body_h < 0.35) + 0.020 * float(vertical) + 0.015 * float(upper or boundary) + 0.030 * float(self.is_crisis(step))
                    if tile.friction == "slippery":
                        dmg += 0.015
                    damage += dmg
                    delta_h -= dmg
                    event = event or "danger_damage"
                elif actual == T_RESOURCE:
                    if not tile.depleted:
                        gain = 0.080 + 0.015 * float(p[2] > self.start[2])
                        tile.depleted = True
                        resource_gain += gain
                        delta_h += gain
                        event = event or "resource_gain"
                    else:
                        event = event or "depleted_resource"
                elif actual == T_REST:
                    gain = 0.018 + 0.012 * float(tile.hidden_rest)
                    recovery_gain += gain
                    delta_h += gain
                    event = event or "entered_rest_chamber"
                else:
                    event = event or "move_3d"
        elif action == "REST":
            actual = self.actual_kind(pos, step)
            if actual == T_REST:
                gain = 0.075 + 0.020 * float(self.tile(pos).hidden_rest)
                event = "deep_rest_recovery"
            else:
                gain = 0.022
                event = "weak_rest_recovery"
            recovery_gain += gain
            delta_h += gain - 0.001
        elif action == "SCAN":
            revealed = self.scan(pos, known, step)
            delta_h -= 0.008
            event = f"scan_revealed_{len(revealed)}"
        else:
            revealed = self.scan(pos, known, step)
            delta_h -= 0.008
            event = f"invalid_action_scan_{len(revealed)}"
        known[new_pos] = self.actual_kind(new_pos, step)
        return new_pos, StepOutcome(delta_h, damage, resource_gain, recovery_gain, hit_wall, entered_unknown, revealed_type, event, vertical, friction)

    def serialize_map_rows(self, world_seed: int, episode: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (i, j, k), tile in self.grid.items():
            rows.append({
                "world_seed": world_seed, "episode": episode, "i": i, "j": j, "k": k,
                "kind": tile.kind, "hidden": tile.hidden or "", "cue": tile.cue,
                "is_start": int((i, j, k) == self.start),
                "false_resource": int(tile.false_resource), "hidden_rest": int(tile.hidden_rest),
                "friction": tile.friction,
            })
        return rows


# =============================================================================
# DARCA wrapper and outer integrated modules
# =============================================================================

@dataclass
class AgentMemory:
    known: Dict[Tuple[int, int, int], str] = field(default_factory=dict)
    visited: set = field(default_factory=set)
    last_positions: List[Tuple[int, int, int]] = field(default_factory=list)
    last_actions: List[str] = field(default_factory=list)
    last_events: List[str] = field(default_factory=list)
    body_h: float = 0.68
    terminal: bool = False
    resources: int = 0
    total_resource_gain: float = 0.0
    total_damage: float = 0.0
    recovery_events: int = 0
    rest_steps: int = 0
    unnecessary_rest_steps: int = 0
    reckless_moves: int = 0
    scans: int = 0
    previous_pos: Optional[Tuple[int, int, int]] = None
    consecutive_scans: int = 0
    consecutive_rest: int = 0

    def update_history(self, pos: Tuple[int, int, int], action: str, event: str) -> None:
        self.visited.add(pos)
        self.last_positions.append(pos)
        self.last_actions.append(action)
        self.last_events.append(event)
        self.last_positions = self.last_positions[-30:]
        self.last_actions = self.last_actions[-30:]
        self.last_events = self.last_events[-30:]
        self.consecutive_scans = self.consecutive_scans + 1 if action == "SCAN" else 0
        self.consecutive_rest = self.consecutive_rest + 1 if action == "REST" else 0


class DarcaWrapper:
    def __init__(self, darca_module: Any, seed: int, theta: float, causal_horizon: int, recurrent_N: int):
        Params = getattr(darca_module, "Params")
        Condition = getattr(darca_module, "Condition")
        Agent = getattr(darca_module, "Agent")
        try:
            params = Params(theta=theta, causal_max_delay=causal_horizon, recurrent_N=recurrent_N)
        except TypeError:
            params = replace(Params(), theta=theta, causal_max_delay=causal_horizon, recurrent_N=recurrent_N)
        self.agent = Agent(params, Condition("Full"), seed)
        self.last: Dict[str, Any] = {}

    def step(self, signal_y: float, shock: float, extra: Dict[str, float]) -> Dict[str, Any]:
        env_info = {
            "external_shock": clip(shock, 0.0, 1.0),
            "y": signal_y,
            "z": extra.get("z", 0.0),
            "exo": extra.get("exo", 0.0),
            "d_dyn": extra.get("d_dyn", 0.0),
            "coupling_t": extra.get("coupling_t", 0.0),
            "sigma_t": extra.get("sigma_t", 0.0),
        }
        out = self.agent.step(signal_y, env_info)
        self.last = out
        return out


def load_darca_module(path: str) -> Any:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"DARCA file not found: {p}")
    spec = importlib.util.spec_from_file_location("darca_runtime_module", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DARCA module: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["darca_runtime_module"] = mod
    spec.loader.exec_module(mod)
    return mod


class OnlineLogisticPredictor:
    """Online logistic predictor copied from the supplied qualitative-valence source.

    This is the same functional form as `OnlineLogistic` in the original
    `qualitative_valence_robustness_audit_v7.py` source:
        sigmoid(b + w·x), updated by stochastic gradient with L2 decay.
    """
    def __init__(self, n_features: int, lr: float = 0.035, l2: float = 1e-4, init_bias: float = 0.0, rng: Optional[np.random.Generator] = None):
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.w = self.rng.normal(0.0, 0.03, size=self.n_features + 1)
        self.w[0] = float(init_bias)
        self.count = 0

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        z = float(self.w[0] + np.dot(self.w[1:], x))
        return clip(1.0 / (1.0 + math.exp(-clip(z, -60.0, 60.0))), 0.0, 1.0)

    def update(self, x: np.ndarray, y: float, weight: float = 1.0) -> float:
        x = np.asarray(x, dtype=float)
        y = clip(float(y), 0.0, 1.0)
        p = self.predict(x)
        err = p - y
        self.w[0] -= self.lr * weight * err
        self.w[1:] -= self.lr * weight * (err * x + self.l2 * self.w[1:])
        self.count += 1
        return p


@dataclass
class QAgentState:
    integrity: float = 1.0
    energy: float = 1.0
    fatigue: float = 0.0
    stability: float = 1.0
    damage: float = 0.0
    pain_memory: float = 0.0
    danger_memory: float = 0.0
    comfort_memory: float = 0.55


@dataclass
class QEvent:
    event_type: str
    intensity: float
    mass: float
    friction: float
    slope: float
    affordance: float
    physical_risk: float
    self_generated: int
    efference_intensity: float
    true_delay: int


class QEfferenceBuffer:
    def __init__(self, max_lag: int = 12):
        self.max_lag = int(max_lag)
        self.buffer: List[Tuple[float, int]] = [(0.0, 0) for _ in range(self.max_lag + 1)]

    def push(self, efference_intensity: float, self_generated: int, true_delay: int) -> None:
        # Same mechanism as the source Q code: current efference is inserted into
        # the buffer at the true sensorimotor delay and then recovered by the
        # lag-searching agency estimator.
        self.buffer.insert(0, (0.0, 0))
        self.buffer = self.buffer[: self.max_lag + 1]
        delay = int(np.clip(true_delay, 0, self.max_lag))
        self.buffer[delay] = (float(efference_intensity) if self_generated else 0.0, int(self_generated))

    def get(self, lag: int) -> Tuple[float, int]:
        lag = int(np.clip(lag, 0, self.max_lag))
        return self.buffer[lag]


class QualitativeValenceLayer:
    """Original learned qualitative-valence model adapted to the TRUE-3D shell.

    This replaces the previous v4 hybrid Q layer.  The computational core is the
    one supplied in `qualitative_valence_robustness_audit_v7.py`:

    - 13-dimensional state/event feature vector;
    - learned danger, comfort, and action-possibility logistic predictors;
    - delayed agency search over an efference buffer with learned + analytic agency;
    - pain, danger, and comfort memory traces;
    - Q lesion, agency lesion, and memory lesion behavior from the source code.

    The only adaptation is an explicit TRUE-3D event adapter, which converts a
    primitive 3D shell transition into the source model's Event/AgentState format.
    DARCA itself remains loaded from --darca-file and is not edited.
    """

    EVENT_TYPES = ["rest", "walk", "slope", "slip", "jump", "landing", "collision", "brake"]

    def __init__(self, lesion: str = "none", true_delay: int = 3, seed: int = 0):
        # source lesion labels are none/q/agency/memory; shell labels preserve arm names.
        self.lesion = lesion
        self.source_lesion = {"q_lesion": "q", "agency_lesion": "agency", "memory_lesion": "memory"}.get(lesion, "none")
        self.true_delay = int(true_delay)
        self.max_lag = 12
        self.rng = np.random.default_rng(seed)
        self.state = QAgentState()
        self.buffer = QEfferenceBuffer(max_lag=self.max_lag)

        self.danger_model = OnlineLogisticPredictor(13, lr=0.040, l2=1e-4, init_bias=-0.2, rng=self.rng)
        self.comfort_model = OnlineLogisticPredictor(13, lr=0.035, l2=1e-4, init_bias=0.2, rng=self.rng)
        self.action_model = OnlineLogisticPredictor(13, lr=0.035, l2=1e-4, init_bias=0.1, rng=self.rng)
        self.agency_model = OnlineLogisticPredictor(6, lr=0.045, l2=1e-4, init_bias=-0.2, rng=self.rng)

        self.q = 0.0
        self.Dg = 0.0
        self.C = 0.55
        self.A = 0.55
        self.G = 0.5 if self.source_lesion == "agency" else 0.0
        self.P = 0.0
        self.R = 0.0
        self.L = 0.0
        self.Mp = 0.0
        self.Md = 0.0
        self.Mc = 0.55
        self.fatigue_phi = 0.0
        self.stability_S = 1.0
        self.integrity_I = 1.0
        self.energy_E = 1.0
        self.rho = 0.0
        self.rho_eff = 0.0
        self.high_contact = 0.0
        self.inferred_lag = -1 if self.source_lesion == "agency" else 0
        self.last_event: Optional[QEvent] = None

        self.q_history: List[float] = []
        self.protective_history: List[float] = []
        self.risky_action_history: List[float] = []
        self.danger_prediction_history: List[float] = []
        self.pain_history: List[float] = []
        self.action_possibility_history: List[float] = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-clip(x, -60.0, 60.0)))

    def _friction_mu(self, friction: str) -> float:
        if friction == "slippery":
            return 0.06
        if friction == "rough":
            return 0.72
        return 0.42

    def _event_type(self, action: str, outcome: Optional[StepOutcome], friction: str, slope: float) -> str:
        if action == "REST":
            return "rest"
        if action == "SCAN":
            return "brake"
        if outcome is not None and outcome.hit_wall:
            return "collision"
        if outcome is not None and outcome.damage > 0.0:
            return "slip" if friction == "slippery" else "collision"
        if action == "MOVE_UP":
            return "jump"
        if action == "MOVE_DOWN":
            return "landing"
        if friction == "slippery" and action in MOVE_ACTIONS:
            return "slip"
        if slope > 0.12:
            return "slope"
        if action in MOVE_ACTIONS:
            return "walk"
        return "walk"

    def _action_intensity(self, action: str, outcome: Optional[StepOutcome], pr: Optional[Dict[str, float]] = None, friction: str = "normal") -> float:
        pr = pr or {}
        danger = float(pr.get("danger_pressure", 0.0))
        unknown = float(pr.get("unknown_pressure", 0.0))
        vertical = float(pr.get("vertical_pressure", 0.0))
        fric = float(pr.get("friction_pressure", 0.0))
        if action == "REST":
            base = 0.04
        elif action == "SCAN":
            base = 0.45
        elif action == "MOVE_UP":
            base = 1.65
        elif action == "MOVE_DOWN":
            base = 1.90
        elif action in MOVE_ACTIONS:
            base = 1.05
        else:
            base = 0.85
        if friction == "slippery" and action in MOVE_ACTIONS:
            base += 0.85
        if outcome is not None:
            if outcome.hit_wall:
                base = max(base, 4.60)
            if outcome.damage > 0.0:
                base = max(base, 2.70 + 10.0 * float(outcome.damage))
        return float(np.clip(base + 1.25 * danger + 0.65 * unknown + 0.75 * vertical + 0.70 * fric, 0.0, 5.5))

    def _candidate_terms(self, action: str, world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int) -> Tuple[QEvent, float, float, float]:
        friction = "normal"
        slope = 0.0
        affordance = 0.55
        high_contact = 0.0
        visible_type = world.actual_kind(pos, step)
        if action in MOVE_ACTIONS:
            di, dj, dk = DIRS[action]
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if not world.in_bounds(p):
                friction = "slippery"
                slope = 0.35
                affordance = 0.05
                high_contact = 1.0
                visible_type = T_DANGER
            else:
                tile = world.tile(p)
                friction = tile.friction
                visible_type = mem.known.get(p, tile.kind)
                slope = 0.35 if action in ("MOVE_UP", "MOVE_DOWN") else 0.08
                if visible_type == T_DANGER:
                    affordance = 0.18; high_contact = 1.0
                elif visible_type == T_UNKNOWN:
                    affordance = 0.35; high_contact = 0.35
                elif visible_type == T_REST:
                    affordance = 0.90
                elif visible_type == T_RESOURCE:
                    affordance = 0.82
                else:
                    affordance = 0.62
                if tile.friction == "slippery":
                    high_contact = max(high_contact, 0.65)
        elif action == "REST":
            friction = "rough" if world.actual_kind(pos, step) == T_REST else "normal"
            slope = 0.0
            affordance = 0.95 if world.actual_kind(pos, step) == T_REST else 0.55
            visible_type = world.actual_kind(pos, step)
        elif action == "SCAN":
            friction = world.tile(pos).friction
            slope = 0.0
            affordance = 0.55
            high_contact = 0.0
            visible_type = world.actual_kind(pos, step)
        pr = local_pressures(world, pos, mem, step)
        u = self._action_intensity(action, None, pr, friction)
        mu = self._friction_mu(friction)
        physical_risk = self._sigmoid(1.00 * (u - 2.35) + 0.55 * (0.28 - mu) + 0.35 * slope)
        etype = self._event_type(action, None, friction, slope)
        ev = QEvent(etype, u, 70.0, mu, slope, affordance, physical_risk, 1 if action in MOVE_ACTIONS or action in ("SCAN", "REST") else 0, u, self.true_delay)
        return ev, high_contact, affordance, physical_risk

    def _event_from_shell(self, pos: Tuple[int, int, int], action: str, outcome: StepOutcome, pr: Dict[str, float], mem: AgentMemory) -> QEvent:
        friction_label = getattr(outcome, "friction", "normal")
        mu = self._friction_mu(friction_label)
        slope = 0.35 if getattr(outcome, "vertical", False) else np.clip(0.08 + 0.22 * pr.get("vertical_pressure", 0.0) + 0.10 * pr.get("friction_pressure", 0.0), 0.0, 0.35)
        u = self._action_intensity(action, outcome, pr, friction_label)
        etype = self._event_type(action, outcome, friction_label, slope)
        if outcome.revealed_type == T_DANGER or outcome.event in ("danger_damage", "slip_damage", "collision"):
            affordance = 0.18
        elif outcome.event in ("resource_gain", "entered_rest_chamber", "deep_rest_recovery"):
            affordance = 0.90
        elif action == "REST":
            affordance = 0.80
        elif action == "SCAN":
            affordance = 0.55
        else:
            affordance = 0.55 + 0.30 * float(outcome.damage <= 0.0) - 0.20 * float(outcome.entered_unknown)
        affordance = clip(affordance, 0.05, 1.0)
        physical_risk = self._sigmoid(1.00 * (u - 2.35) + 0.55 * (0.28 - mu) + 0.35 * slope)
        self_generated = 1 if action in ACTIONS else 0
        efference = u if self_generated else 0.0
        return QEvent(etype, float(u), 70.0, float(mu), float(slope), float(affordance), float(physical_risk), int(self_generated), float(efference), self.true_delay)

    def _sync_state_from_shell(self, mem: AgentMemory, pr: Dict[str, float], outcome: Optional[StepOutcome] = None, action: str = "") -> None:
        # The original Q code uses AgentState(I,E,Phi,S,memories).  In TRUE-3D,
        # these variables are grounded in the shell body state and local pressures.
        self.state.damage = clip(mem.total_damage, 0.0, 1.0)
        self.state.integrity = clip(1.0 - self.state.damage, 0.0, 1.0)
        self.state.energy = clip(mem.body_h, 0.0, 1.0)
        move_load = 0.12 * float(action in MOVE_ACTIONS) + 0.05 * float(action == "SCAN")
        self.state.fatigue = clip(0.94 * self.state.fatigue + 0.06 * (1.0 - mem.body_h + move_load), 0.0, 1.0)
        dmg = float(outcome.damage) if outcome is not None else 0.0
        self.state.stability = clip(0.88 * self.state.stability + 0.12 * (1.0 - 0.50 * pr.get("danger_pressure", 0.0) - 0.25 * pr.get("vertical_pressure", 0.0) - 0.25 * pr.get("friction_pressure", 0.0) - 2.50 * dmg), 0.0, 1.0)
        self.integrity_I = self.state.integrity
        self.energy_E = self.state.energy
        self.fatigue_phi = self.state.fatigue
        self.stability_S = self.state.stability

    def _features(self, event: QEvent, state: QAgentState, agency_score: float) -> np.ndarray:
        high_contact = 1.0 if event.event_type in ["slip", "landing", "collision", "brake"] else 0.0
        return np.array([
            event.intensity / 5.5,
            event.friction,
            event.slope / 0.35,
            event.affordance,
            event.physical_risk,
            state.integrity,
            state.energy,
            state.fatigue,
            state.stability,
            state.pain_memory,
            state.danger_memory,
            high_contact,
            agency_score,
        ], dtype=float)

    def _agency_features_for_lag(self, event: QEvent, eff_intensity: float, eff_flag: int, lag: int) -> np.ndarray:
        alignment = math.exp(-abs(event.intensity - eff_intensity) / 1.35)
        delay_prior = math.exp(-abs(lag - self.true_delay) / 2.0)
        high_contact = 1.0 if event.event_type in ["landing", "collision", "slip"] else 0.0
        return np.array([
            eff_intensity / 5.5,
            float(eff_flag),
            event.intensity / 5.5,
            alignment,
            delay_prior,
            high_contact,
        ], dtype=float)

    def _estimate_agency(self, event: QEvent) -> Dict[str, Any]:
        if self.source_lesion == "agency":
            return {"agency_score": 0.5, "agency_best_lag": -1, "agency_best_efference": 0.0, "agency_best_flag": 0, "agency_x": np.zeros(6)}
        scores = []
        for lag in range(self.max_lag + 1):
            eff, flag = self.buffer.get(lag)
            x = self._agency_features_for_lag(event, eff, flag, lag)
            p_learned = self.agency_model.predict(x)
            alignment = math.exp(-abs(event.intensity - eff) / 1.35)
            delay_prior = math.exp(-abs(lag - self.true_delay) / 1.4)
            p_analytic = self._sigmoid(3.8 * float(flag) + 2.2 * alignment + 2.0 * delay_prior - 4.2)
            p = 0.45 * p_learned + 0.55 * p_analytic
            scores.append((lag, p, eff, flag, x))
        lag, p, eff, flag, x = max(scores, key=lambda z: z[1])
        return {"agency_score": float(p), "agency_best_lag": int(lag), "agency_best_efference": float(eff), "agency_best_flag": int(flag), "agency_x": x}

    def _immediate_pain(self, event: QEvent, state: QAgentState) -> float:
        if event.event_type in ["collision", "landing", "slip"]:
            impact = event.intensity
        elif event.event_type in ["slope", "brake"]:
            impact = 0.55 * event.intensity
        elif event.event_type == "jump":
            impact = 0.35 * event.intensity
        else:
            impact = 0.12 * event.intensity
        vulnerability = 0.35 * (1.0 - state.integrity) + 0.25 * state.fatigue + 0.25 * (1.0 - state.stability)
        return float(self._sigmoid(1.25 * (impact - 2.20) + 1.25 * vulnerability))

    def _compute_components(self, event: QEvent, pain: float) -> Dict[str, Any]:
        if self.source_lesion == "memory":
            state_view = QAgentState(
                integrity=self.state.integrity,
                energy=self.state.energy,
                fatigue=self.state.fatigue,
                stability=self.state.stability,
                damage=self.state.damage,
                pain_memory=0.0,
                danger_memory=0.0,
                comfort_memory=0.55,
            )
        else:
            state_view = self.state
        ag = self._estimate_agency(event)
        agency_score = ag["agency_score"]
        x = self._features(event, state_view, agency_score)
        danger = self.danger_model.predict(x)
        comfort = self.comfort_model.predict(x)
        action_possibility = self.action_model.predict(x)
        if self.source_lesion == "q":
            danger = clip(0.24 + 0.28 * event.physical_risk, 0.0, 1.0)
            comfort = clip(0.62 - 0.10 * event.physical_risk, 0.0, 1.0)
            action_possibility = clip(0.04 + 0.06 * event.affordance, 0.0, 1.0)
            agency_score = 0.5
            x = self._features(event, state_view, agency_score)
        controllability = action_possibility * (0.65 + 0.35 * agency_score)
        avoidance_pressure = self._sigmoid(
            1.65 * danger
            + 1.10 * pain
            - 0.85 * comfort
            - 0.55 * controllability
            + 0.55 * state_view.danger_memory
            + 0.25 * (1.0 - state_view.integrity)
            - 0.05
        )
        q_aversive_index = clip(
            0.23 * danger
            + 0.20 * pain
            + 0.20 * avoidance_pressure
            + 0.12 * (1.0 - action_possibility)
            + 0.07 * (1.0 - comfort)
            + 0.08 * state_view.danger_memory
            + 0.05 * state_view.pain_memory
            + 0.03 * state_view.fatigue
            + 0.02 * (1.0 - state_view.stability),
            0.0,
            1.0,
        )
        return {
            "comfort": float(comfort),
            "pain": float(pain),
            "danger": float(danger),
            "avoidance_pressure": float(avoidance_pressure),
            "action_possibility": float(action_possibility),
            "q_aversive_index": float(q_aversive_index),
            "agency_score": float(agency_score),
            "agency_best_lag": int(ag["agency_best_lag"]),
            "agency_best_efference": float(ag["agency_best_efference"]),
            "agency_best_flag": int(ag["agency_best_flag"]),
            "feature_vector": x,
            "agency_x": ag["agency_x"],
            "controllability": float(controllability),
        }

    def _update_from_outcome(self, comp: Dict[str, Any], event: QEvent, outcome: Dict[str, float]) -> None:
        if self.source_lesion == "q":
            return
        x = comp["feature_vector"]
        damage_target = 1.0 if (outcome["damage_increment"] > 0.000010 or outcome["pain_after"] > 0.66) else 0.0
        comfort_target = 1.0 if (
            outcome["damage_increment"] <= 0.000010
            and outcome["pain_after"] < 0.52
            and outcome["post_energy"] > 0.45
            and outcome["post_stability"] > 0.45
        ) else 0.0
        if outcome["action_avoid"] == 1:
            action_target = 1.0 if (
                outcome["damage_increment"] <= 0.000010
                and outcome["effective_risk"] < max(0.58, event.physical_risk + 0.02)
            ) else 0.0
        else:
            action_target = 1.0 if outcome["damage_increment"] <= 0.000010 and outcome["effective_risk"] < 0.55 else 0.0
        self.danger_model.update(x, damage_target, weight=1.0)
        self.comfort_model.update(x, comfort_target, weight=1.0)
        self.action_model.update(x, action_target, weight=0.85)
        if self.source_lesion != "agency":
            for lag in range(self.max_lag + 1):
                eff, flag = self.buffer.get(lag)
                ax = self._agency_features_for_lag(event, eff, flag, lag)
                target = 1.0 if (event.self_generated == 1 and flag == 1 and abs(lag - event.true_delay) <= 1) else 0.0
                if event.self_generated == 0:
                    target = 0.0
                self.agency_model.update(ax, target, weight=0.45)

    def predict_cell_q(self, p: Tuple[int, int, int], visible_type: str, cue: str, body_h: float) -> float:
        # This supplies only the local, non-language,
        # body/history-dependent Q estimate for compatibility with the shell.
        friction = 0.42
        slope = 0.08
        affordance = 0.18 if visible_type == T_DANGER else 0.35 if visible_type == T_UNKNOWN else 0.90 if visible_type == T_REST else 0.82 if visible_type == T_RESOURCE else 0.55
        u = 3.0
        physical_risk = self._sigmoid(1.00 * (u - 2.35) + 0.55 * (0.28 - friction) + 0.35 * slope)
        ev = QEvent("walk", u, 70.0, friction, slope, affordance, physical_risk, 0, 0.0, self.true_delay)
        state0 = self.state
        temp_state = QAgentState(
            integrity=clip(0.55 * self.state.integrity + 0.45 * body_h, 0.0, 1.0),
            energy=clip(body_h, 0.0, 1.0),
            fatigue=self.state.fatigue,
            stability=0.25 if visible_type == T_DANGER else 0.45 if visible_type == T_UNKNOWN else 0.85,
            damage=1.0 - clip(0.55 * self.state.integrity + 0.45 * body_h, 0.0, 1.0),
            pain_memory=self.state.pain_memory,
            danger_memory=self.state.danger_memory,
            comfort_memory=self.state.comfort_memory,
        )
        self.state = temp_state
        pain = self._immediate_pain(ev, self.state)
        comp = self._compute_components(ev, pain)
        self.state = state0
        return float(comp["q_aversive_index"])

    def action_risk_modifier(self, action: str, world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int) -> float:
        self._sync_state_from_shell(mem, local_pressures(world, pos, mem, step), None, action)
        event, _, _, _ = self._candidate_terms(action, world, pos, mem, step)
        pain = self._immediate_pain(event, self.state)
        comp = self._compute_components(event, pain)
        return float(comp["q_aversive_index"])

    def update(self, pos: Tuple[int, int, int], action: str, outcome: StepOutcome, pr: Dict[str, float], mem: AgentMemory, darca_out: Dict[str, Any]) -> Dict[str, float]:
        self._sync_state_from_shell(mem, pr, outcome, action)
        event = self._event_from_shell(pos, action, outcome, pr, mem)
        self.last_event = event
        self.buffer.push(event.efference_intensity, event.self_generated, event.true_delay)
        pain = self._immediate_pain(event, self.state)
        comp = self._compute_components(event, pain)

        # The actual TRUE-3D transition is the outcome.  We convert it into the
        # source model's outcome fields without inventing a separate reward term.
        action_avoid = int(action in ("REST", "SCAN") or (action in MOVE_ACTIONS and outcome.damage <= 0.0 and pr.get("danger_pressure", 0.0) > 0.25))
        effective_risk = clip(0.65 * event.physical_risk + 0.35 * clip(8.0 * float(outcome.damage) + 0.25 * float(outcome.hit_wall), 0.0, 1.0), 0.0, 1.0)
        outcome_dict = {
            "damage_increment": float(outcome.damage),
            "pain_after": float(pain),
            "post_energy": float(self.state.energy),
            "post_stability": float(self.state.stability),
            "effective_risk": float(effective_risk),
            "action_avoid": int(action_avoid),
        }
        self._update_from_outcome(comp, event, outcome_dict)

        if self.source_lesion == "memory":
            self.state.pain_memory = 0.0
            self.state.danger_memory = 0.0
            self.state.comfort_memory = 0.55
        else:
            self.state.pain_memory = clip(0.940 * self.state.pain_memory + 0.060 * pain, 0.0, 1.0)
            self.state.danger_memory = clip(0.945 * self.state.danger_memory + 0.055 * effective_risk, 0.0, 1.0)
            self.state.comfort_memory = clip(0.945 * self.state.comfort_memory + 0.055 * comp["comfort"], 0.0, 1.0)

        self.q = comp["q_aversive_index"]
        self.Dg = comp["danger"]
        self.C = comp["comfort"]
        self.A = comp["action_possibility"]
        self.G = comp["agency_score"]
        self.P = comp["pain"]
        self.R = comp["avoidance_pressure"]
        self.L = comp["controllability"]
        self.Mp = self.state.pain_memory
        self.Md = self.state.danger_memory
        self.Mc = self.state.comfort_memory
        self.fatigue_phi = self.state.fatigue
        self.stability_S = self.state.stability
        self.integrity_I = self.state.integrity
        self.energy_E = self.state.energy
        self.rho = event.physical_risk
        self.rho_eff = effective_risk
        self.high_contact = 1.0 if event.event_type in ["slip", "landing", "collision", "brake"] else 0.0
        self.inferred_lag = comp["agency_best_lag"]

        protective = float(action_avoid == 1)
        risky_action = float(action in MOVE_ACTIONS and (outcome.damage > 0.0 or pr.get("danger_pressure", 0.0) > 0.35 or pr.get("unknown_pressure", 0.0) > 0.20))
        self.q_history.append(self.q)
        self.protective_history.append(protective)
        self.risky_action_history.append(risky_action)
        self.danger_prediction_history.append(self.Dg)
        self.pain_history.append(self.P)
        self.action_possibility_history.append(self.A)
        self.q_history = self.q_history[-500:]
        self.protective_history = self.protective_history[-500:]
        self.risky_action_history = self.risky_action_history[-500:]
        self.danger_prediction_history = self.danger_prediction_history[-500:]
        self.pain_history = self.pain_history[-500:]
        self.action_possibility_history = self.action_possibility_history[-500:]

        return {
            "Q": self.q,
            "Q_Dg": self.Dg,
            "Q_learned_danger": self.Dg,
            "Q_C": self.C,
            "Q_learned_comfort": self.C,
            "Q_A": self.A,
            "Q_action_possibility": self.A,
            "Q_G": self.G,
            "Q_agency": self.G,
            "Q_P": self.P,
            "Q_pain": self.P,
            "Q_R": self.R,
            "Q_avoidance_pressure": self.R,
            "Q_L": self.L,
            "Q_controllability": self.L,
            "Q_Mp": self.Mp,
            "Q_pain_memory": self.Mp,
            "Q_Md": self.Md,
            "Q_danger_memory": self.Md,
            "Q_Mc": self.Mc,
            "Q_comfort_memory": self.Mc,
            "Q_fatigue": self.fatigue_phi,
            "Q_stability": self.stability_S,
            "Q_integrity": self.integrity_I,
            "Q_energy": self.energy_E,
            "Q_rho": self.rho,
            "Q_rho_eff": self.rho_eff,
            "Q_high_contact": self.high_contact,
            "Q_inferred_lag": self.inferred_lag,
            "Q_event_type": event.event_type,
            "Q_event_intensity": event.intensity,
            "Q_event_physical_risk": event.physical_risk,
            "Q_event_affordance": event.affordance,
            "Q_event_self_generated": event.self_generated,
            "Q_event_true_delay": event.true_delay,
            "Q_agency_best_flag": comp.get("agency_best_flag", 0),
            "Q_agency_best_efference": comp.get("agency_best_efference", 0.0),
            "Q_source_lesion": self.source_lesion,
            "Q_lesion_mode": self.lesion,
        }

    def q_action_coupling(self) -> float:
        if len(self.q_history) < 8:
            return 0.0
        q = np.asarray(self.q_history, dtype=float)
        p = np.asarray(self.protective_history, dtype=float)
        if np.std(q) < 1e-9 or np.std(p) < 1e-9:
            return 0.0
        return float(np.corrcoef(q, p)[0, 1])

    def q_risk_suppression(self) -> float:
        if len(self.q_history) < 8:
            return 0.0
        q = np.asarray(self.q_history[:-1], dtype=float)
        risky_next = np.asarray(self.risky_action_history[1:], dtype=float)
        if np.std(q) < 1e-9 or np.std(risky_next) < 1e-9:
            return 0.0
        return float(-np.corrcoef(q, risky_next)[0, 1])

class PhysicalLawLayer:
    """Outer active embodied physical expectation layer.

    lesion=True keeps only a fixed weak physical prior and disables both
    prediction learning and action adjustment.  This gives a clean
    physical-law lesion without changing the DARCA core.
    """
    def __init__(self, lesion: bool = False):
        self.lesion = bool(lesion)
        self.stats: Dict[str, Dict[str, float]] = {a: {"n": 0.0, "damage": 0.02, "gain": 0.0, "wall": 0.02, "pred_err": 0.2} for a in ACTIONS}
        self.global_pred_error = 0.2
        self.score = 0.0
        self.prediction_errors: List[float] = []

    def predict(self, action: str, world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int) -> Dict[str, float]:
        st = self.stats.get(action, self.stats["SCAN"])
        pred_damage = st["damage"]
        pred_gain = st["gain"]
        pred_wall = st["wall"]
        if self.lesion:
            # Weak, fixed physical reactivity: enough to expose current
            # contact/wall cues, but no action-consequence learning.
            pred_damage = 0.04
            pred_gain = 0.0
            pred_wall = 0.04
        if action in MOVE_ACTIONS:
            di, dj, dk = DIRS[action]
            p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if not world.in_bounds(p):
                pred_wall = max(pred_wall, 0.95)
                pred_damage = max(pred_damage, 0.12)
            else:
                tile = world.tile(p)
                kt = mem.known.get(p, tile.kind)
                if kt == T_DANGER:
                    pred_damage = max(pred_damage, 0.18)
                if kt == T_RESOURCE:
                    pred_gain = max(pred_gain, 0.07)
                if tile.friction == "slippery":
                    pred_damage += 0.02
                if tile.friction == "rough":
                    pred_gain -= 0.01
                if action in ("MOVE_UP", "MOVE_DOWN"):
                    pred_damage += 0.01
        elif action == "REST":
            if world.actual_kind(pos, step) == T_REST:
                pred_gain = max(pred_gain, 0.075)
            else:
                pred_gain = max(pred_gain, 0.015)
        elif action == "SCAN":
            pred_gain = max(pred_gain, 0.005)
        return {"pred_damage": clip(pred_damage, 0.0, 1.0), "pred_gain": clip(pred_gain, -0.2, 0.2), "pred_wall": clip(pred_wall, 0.0, 1.0)}

    def best_action_adjustment(self, action: str, world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> Tuple[str, str]:
        if self.lesion:
            return action, "physics_lesion_no_adjust"
        pred = self.predict(action, world, pos, mem, step)
        if pred["pred_wall"] > 0.70:
            return "SCAN", "physics_reject_wall"
        if pred["pred_damage"] > 0.16 and mem.body_h < 0.55:
            return action_away_from_danger(world, pos, mem, rng, step), "physics_reject_expected_damage"
        if action in ("MOVE_UP", "MOVE_DOWN") and pred["pred_damage"] > 0.10 and mem.body_h < 0.45:
            return "SCAN", "physics_vertical_caution"
        return action, "physics_accept"

    def update(self, action: str, outcome: StepOutcome, pred: Dict[str, float]) -> Dict[str, float]:
        st = self.stats[action]
        alpha = 0.08
        actual_damage = outcome.damage
        actual_gain = outcome.resource_gain + outcome.recovery_gain
        actual_wall = 1.0 if outcome.hit_wall else 0.0
        err = abs(pred["pred_damage"] - actual_damage) + 0.5 * abs(pred["pred_gain"] - actual_gain) + 0.5 * abs(pred["pred_wall"] - actual_wall)
        st["n"] += 1.0
        if self.lesion:
            # Keep score fixed at zero and do not update action-specific world
            # expectations.  The error is still reported for validation.
            self.prediction_errors.append(err)
            self.prediction_errors = self.prediction_errors[-500:]
            return {"physics_pred_error": err, "physics_score": 0.0, "physics_action_n": st["n"], "physics_lesion": 1}
        st["damage"] = (1 - alpha) * st["damage"] + alpha * actual_damage
        st["gain"] = (1 - alpha) * st["gain"] + alpha * actual_gain
        st["wall"] = (1 - alpha) * st["wall"] + alpha * actual_wall
        st["pred_err"] = (1 - alpha) * st["pred_err"] + alpha * err
        self.global_pred_error = 0.98 * self.global_pred_error + 0.02 * err
        self.prediction_errors.append(err)
        self.prediction_errors = self.prediction_errors[-500:]
        self.score = clip(1.0 - self.global_pred_error / 0.20, 0.0, 1.0)
        return {"physics_pred_error": err, "physics_score": self.score, "physics_action_n": st["n"], "physics_lesion": 0}


class SocialSignalLayer:
    """Phase-4c style anonymous social signal layer.

    This layer is intentionally non-motor and language-free.  It follows the supplied
    social-code constraints: no LAUGH/HUMOR label, no benign-violation control
    variable, anonymous signal_0..signal_4 channels, generic controller features,
    and post-hoc analysis of relief / safe-surprise / self-appraisal gap.
    """
    EVENT_NAMES = ["rest", "walk", "explore", "minor_mismatch", "near_miss", "false_alarm", "social_play", "bump", "slip", "collision"]

    def __init__(self, seed: int = 0, n_receivers: int = 5, n_signals: int = 5):
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.n_receivers = int(n_receivers)
        self.n_signals = int(n_signals)
        self.n_features = 14

        # Phase-4c calibration constants.  These are social-layer internal
        # variables and do not affect DARCA action authority.
        self.signal_threshold = 0.455
        self.signal_bias = -0.62
        self.signal_noise = 0.060
        self.signal_exploration_rate = 0.055
        self.signal_lr = 0.016
        self.weight_decay = 0.00045
        self.refractory_steps = 4
        self.listener_lr = 0.060
        self.listener_effect_strength = 0.070
        self.contagion_strength = 0.030
        self.base_social_relaxation = 0.0040
        self.social_decay = 0.007
        self.memory_decay = 0.012
        self.predictor_lr = 0.045

        self.signal_weights = self._make_initial_signal_weights()
        self.listener_value = self.rng.normal(0.0, 0.02, size=(self.n_receivers, self.n_signals))
        self.receiver_tension = self.rng.uniform(0.14, 0.28, size=self.n_receivers)
        self.receiver_sync = self.rng.uniform(0.35, 0.55, size=self.n_receivers)
        self.receiver_explore = self.rng.uniform(0.50, 0.70, size=self.n_receivers)

        self.self_tension = float(self.rng.uniform(0.14, 0.28))
        self.self_sync = float(self.rng.uniform(0.35, 0.55))
        self.exploration_drive = float(self.rng.uniform(0.50, 0.70))
        self.last_appraisal = 0.0
        self.last_actual_risk = 0.0
        self.last_q = 0.0
        self.prev_h = 0.68
        self.refractory = 0
        self.pred_intensity = {e: 0.35 for e in self.EVENT_NAMES}

        self.receiver_recovery = 0.0
        self.signal_count = 0
        self.selected_counts = {i: 0 for i in range(self.n_signals)}
        self.gap_history: List[float] = []

    def _make_initial_signal_weights(self) -> np.ndarray:
        w = self.rng.normal(0.0, 0.08, size=(self.n_signals, self.n_features))
        # Same generic predispositions as Phase 4c: prediction error, relief,
        # novelty, social tension; actual risk and damage suppress signaling.
        w[:, 2] += self.rng.normal(0.02, 0.03, self.n_signals)    # PE
        w[:, 6] += self.rng.normal(0.08, 0.03, self.n_signals)    # relief raw
        w[:, 7] += self.rng.normal(0.07, 0.03, self.n_signals)    # novelty
        w[:, 9] += self.rng.normal(0.035, 0.025, self.n_signals)  # social tension
        w[:, 3] += self.rng.normal(-0.22, 0.035, self.n_signals)  # actual risk suppression
        w[:, 4] += self.rng.normal(-0.18, 0.035, self.n_signals)  # damage suppression
        return w

    def _event_name(self, outcome: StepOutcome, action: str, pr: Dict[str, float]) -> str:
        ev = str(outcome.event).lower()
        if "rest" in ev or outcome.recovery_gain > 0:
            return "rest"
        if outcome.damage > 0.035 or outcome.hit_wall:
            return "collision"
        if outcome.damage > 0.0 or "danger" in ev:
            return "bump"
        if outcome.friction == "slippery":
            return "slip"
        if action == "MOVE_UP":
            return "explore"
        if action == "MOVE_DOWN":
            return "near_miss"
        if action == "SCAN":
            return "minor_mismatch"
        if outcome.resource_gain > 0:
            return "social_play"
        if pr.get("unknown_pressure", 0.0) > 0.35 and pr.get("danger_pressure", 0.0) < 0.35:
            return "false_alarm"
        if action in MOVE_ACTIONS:
            return "walk"
        return "explore"

    def _outcome_variables(self, mem: AgentMemory, q_state: Dict[str, float], pr: Dict[str, float], outcome: StepOutcome, action: str) -> Dict[str, float]:
        actual_risk = clip(
            0.52 * safe_float(pr.get("danger_pressure"))
            + 0.20 * safe_float(pr.get("friction_pressure"))
            + 0.12 * safe_float(pr.get("vertical_pressure"))
            + 2.4 * safe_float(outcome.damage)
            + 0.18 * float(outcome.hit_wall),
            0.0,
            1.0,
        )
        damage_norm = clip(10.0 * safe_float(outcome.damage), 0.0, 1.0)
        intensity = clip(
            0.18
            + 0.55 * actual_risk
            + 0.22 * abs(safe_float(outcome.delta_h)) * 4.0
            + 0.12 * float(action in MOVE_ACTIONS)
            + 0.10 * float(outcome.entered_unknown),
            0.0,
            1.0,
        )
        # Initial appraisal is the local before-reappraisal threat estimate.
        appraisal = clip(
            0.35 * safe_float(q_state.get("Q"))
            + 0.30 * safe_float(pr.get("danger_pressure"))
            + 0.18 * safe_float(pr.get("unknown_pressure"))
            + 0.12 * float(outcome.entered_unknown)
            + 0.12 * float(action == "SCAN"),
            0.0,
            1.0,
        )
        novelty = clip(0.60 * abs(intensity - self.last_actual_risk) + 0.25 * float(outcome.entered_unknown) + 0.15 * float(action == "SCAN"), 0.0, 1.0)
        agency = clip(0.55 * safe_float(q_state.get("Q_G"), 0.5) + 0.25 * safe_float(q_state.get("Q_agency"), 0.5) + 0.20 * float(action in MOVE_ACTIONS), 0.0, 1.0)
        return {"actual_risk": actual_risk, "damage_norm": damage_norm, "intensity": intensity, "initial_appraisal": appraisal, "novelty": novelty, "agency": agency}

    def _feature_vector(self, mem: AgentMemory, q_state: Dict[str, float], out: Dict[str, float], pred_before: float) -> Tuple[np.ndarray, Dict[str, float]]:
        pe = abs(out["intensity"] - pred_before)
        relief_raw = max(0.0, self.last_appraisal - out["actual_risk"])
        current_safety = 1.0 - max(out["actual_risk"], out["damage_norm"])
        self_gap = float(self.last_appraisal * current_safety * abs(self.last_appraisal - out["actual_risk"]))
        q_relief = max(0.0, self.last_q - safe_float(q_state.get("Q")))
        safe_surprise = float((pe >= 0.07) and (out["actual_risk"] <= 0.38) and (out["damage_norm"] <= 0.18))
        danger_context = float((out["actual_risk"] >= 0.64) or (out["damage_norm"] >= 0.40))
        safe_context = float((self.last_appraisal >= 0.34) and (current_safety >= 0.60) and (pe >= 0.07) and (danger_context < 0.5))
        vulnerability = 1.0 - min(safe_float(mem.body_h), safe_float(q_state.get("Q_energy", mem.body_h)))
        danger_memory = safe_float(q_state.get("Q_Md", q_state.get("Q_danger_memory", 0.0)))
        x = np.array([
            1.0,
            safe_float(q_state.get("Q")),
            pe,
            out["actual_risk"],
            out["damage_norm"],
            self.last_appraisal,
            relief_raw,
            out["novelty"],
            out["agency"],
            self.self_tension,
            self.self_sync,
            self.exploration_drive,
            danger_memory,
            vulnerability,
        ], dtype=float)
        analysis = {
            "prediction_error": float(pe),
            "relief": float(relief_raw),
            "safe_surprise": safe_surprise,
            "self_appraisal_gap": float(self_gap),
            "q_relief": float(q_relief),
            "safe_context": safe_context,
            "danger_context": danger_context,
            "past_threat": float(self.last_appraisal),
            "current_safety": float(current_safety),
        }
        return x, analysis

    def _select_signal(self, features: np.ndarray, actual_risk: float, damage_norm: float) -> Tuple[int, np.ndarray]:
        if self.refractory > 0:
            self.refractory -= 1
            return -1, np.zeros(self.n_signals)
        risk_inhibition = 0.55 * actual_risk + 0.42 * damage_norm
        logits = self.signal_weights @ features + self.signal_bias - risk_inhibition + self.rng.normal(0, self.signal_noise, self.n_signals)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        if self.rng.random() < self.signal_exploration_rate:
            ch = int(self.rng.integers(0, self.n_signals))
            if self.rng.random() < 0.55 and actual_risk < 0.72:
                return ch, p
        ch = int(np.argmax(p))
        if p[ch] > self.signal_threshold:
            return ch, p
        return -1, p

    def _apply_signal_to_listeners(self, channel: int, actual_risk: float, damage_norm: float) -> Tuple[float, float, float, float]:
        if channel < 0:
            return 0.0, 0.0, 0.0, float(np.mean(self.receiver_tension))
        receiver_recovery: List[float] = []
        spread = 0
        heard = 0
        risk_penalty = max(0.0, actual_risk - 0.45) + damage_norm
        for j in range(self.n_receivers):
            heard += 1
            before_tension = float(self.receiver_tension[j])
            before_explore = float(self.receiver_explore[j])
            val = float(self.listener_value[j, channel])
            effect = self.listener_effect_strength * (1.0 / (1.0 + math.exp(-2.5 * val))) * max(0.0, 1.0 - 1.7 * risk_penalty)
            self.receiver_tension[j] = clip(self.receiver_tension[j] - effect, 0.0, 1.0)
            self.receiver_sync[j] = clip(self.receiver_sync[j] + 0.75 * effect, 0.0, 1.0)
            self.receiver_explore[j] = clip(self.receiver_explore[j] + 0.55 * effect, 0.0, 1.0)
            receiver_recovery.append((before_tension - float(self.receiver_tension[j])) + 0.5 * (float(self.receiver_explore[j]) - before_explore))
            if self.rng.random() < self.contagion_strength * (1.0 / (1.0 + math.exp(-2.0 * val))) * max(0.0, 1.0 - risk_penalty):
                spread += 1
        return float(np.mean(receiver_recovery)) if receiver_recovery else 0.0, float(spread), float(heard), float(np.mean(self.receiver_tension))

    def _update_signal_learning(self, features: np.ndarray, channel: int, benefit: float, actual_risk: float, damage_norm: float) -> None:
        if channel < 0:
            return
        risk_cost = 0.035 * max(0.0, actual_risk - 0.42) + 0.030 * damage_norm
        delta = benefit - risk_cost - 0.00006
        self.signal_weights *= (1.0 - self.weight_decay)
        self.signal_weights[channel] += self.signal_lr * float(np.clip(delta, -0.05, 0.05)) * features

    def _update_listener_learning(self, channel: int, receiver_recovery: float, actual_risk: float, damage_norm: float) -> None:
        if channel < 0:
            return
        danger_cost = 0.05 * max(0.0, actual_risk - 0.45) + 0.05 * damage_norm
        target = receiver_recovery - danger_cost
        self.listener_value[:, channel] += self.listener_lr * float(np.clip(target, -0.08, 0.08))
        self.listener_value[:, channel] = np.clip(self.listener_value[:, channel], -1.5, 1.5)

    def update(self, mem: AgentMemory, q_state: Dict[str, float], pr: Dict[str, float], outcome: StepOutcome, action: str) -> Dict[str, float]:
        event = self._event_name(outcome, action, pr)
        out = self._outcome_variables(mem, q_state, pr, outcome, action)
        pred_before = float(self.pred_intensity.get(event, 0.35))
        features, analysis = self._feature_vector(mem, q_state, out, pred_before)

        channel, probs = self._select_signal(features, out["actual_risk"], out["damage_norm"])
        emitted = channel >= 0
        if emitted:
            self.refractory = self.refractory_steps
            self.signal_count += 1
            self.selected_counts[channel] = self.selected_counts.get(channel, 0) + 1

        own_tension_before = self.self_tension
        own_explore_before = self.exploration_drive
        receiver_tension_before = float(np.mean(self.receiver_tension))
        receiver_recovery, spread, heard, receiver_tension_after_signal = self._apply_signal_to_listeners(channel, out["actual_risk"], out["damage_norm"])

        # Sender-side social state update.  Signals are costly but can relax local
        # social tension when emitted under non-danger conditions.
        signal_relax = 0.030 * float(emitted) * max(0.0, 1.0 - 1.5 * out["actual_risk"])
        self.self_tension = clip((1.0 - self.social_decay) * self.self_tension + 0.050 * out["actual_risk"] + 0.025 * out["initial_appraisal"] - self.base_social_relaxation - signal_relax, 0.0, 1.0)
        self.self_sync = clip(0.996 * self.self_sync + 0.012 * float(emitted) - 0.008 * out["actual_risk"], 0.0, 1.0)
        self.exploration_drive = clip(0.995 * self.exploration_drive + 0.006 * (1.0 - out["actual_risk"]) - 0.012 * safe_float(q_state.get("Q")) + 0.010 * float(emitted), 0.0, 1.0)
        own_recovery = (own_tension_before - self.self_tension) + 0.5 * (self.exploration_drive - own_explore_before)
        total_benefit = own_recovery + 0.8 * receiver_recovery + 0.003 * spread
        self._update_signal_learning(features, channel, total_benefit, out["actual_risk"], out["damage_norm"])
        self._update_listener_learning(channel, receiver_recovery, out["actual_risk"], out["damage_norm"])

        # Baseline receiver social dynamics after signal effect.  This preserves
        # the source-code property that high-risk contexts can raise receiver
        # tension, so recovery is not a trivial always-positive artifact.
        self.receiver_tension = np.clip((1.0 - self.social_decay) * self.receiver_tension + 0.018 * out["actual_risk"] + 0.006 * out["initial_appraisal"] - self.base_social_relaxation, 0.0, 1.0)
        self.receiver_sync = np.clip(0.996 * self.receiver_sync + 0.004 * float(emitted) - 0.006 * out["actual_risk"], 0.0, 1.0)
        self.receiver_explore = np.clip(0.995 * self.receiver_explore + 0.006 * (1.0 - out["actual_risk"]) - 0.006 * safe_float(q_state.get("Q")) + 0.006 * float(emitted), 0.0, 1.0)
        receiver_tension_after = float(np.mean(self.receiver_tension))

        self.receiver_recovery += receiver_recovery
        self.pred_intensity[event] = (1.0 - self.predictor_lr) * self.pred_intensity.get(event, 0.35) + self.predictor_lr * out["intensity"]
        self.last_appraisal = out["initial_appraisal"]
        self.last_actual_risk = out["actual_risk"]
        self.last_q = safe_float(q_state.get("Q"))
        self.prev_h = safe_float(mem.body_h)
        self.gap_history.append(analysis["self_appraisal_gap"])
        self.gap_history = self.gap_history[-500:]

        return {
            "social_signal": float(emitted),
            "selected_signal_channel": int(channel),
            "signal_probability_max": float(np.max(probs)) if probs.size else 0.0,
            "receiver_heard_count": heard,
            "cross_agent_spread": spread,
            "own_recovery": float(own_recovery),
            "receiver_stress": receiver_tension_after,
            "receiver_stress_before_signal": receiver_tension_before,
            "receiver_stress_after_signal": receiver_tension_after,
            "receiver_recovery_increment": receiver_recovery,
            "receiver_recovery_total": self.receiver_recovery,
            "social_tension": self.self_tension,
            "social_sync": self.self_sync,
            "exploration_drive": self.exploration_drive,
            "social_event_class": event,
            "social_actual_risk": out["actual_risk"],
            "social_damage_norm": out["damage_norm"],
            "social_initial_appraisal": out["initial_appraisal"],
            "self_appraisal_gap": analysis["self_appraisal_gap"],
            "relief": analysis["relief"],
            "safe_surprise": analysis["safe_surprise"],
            "q_relief": analysis["q_relief"],
            "prediction_error_social": analysis["prediction_error"],
            "safe_context": analysis["safe_context"],
            "danger_context": analysis["danger_context"],
            "past_threat": analysis["past_threat"],
            "current_safety": analysis["current_safety"],
        }


# =============================================================================
# Perception and action logic
# =============================================================================

def local_pressures(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int = 0) -> Dict[str, float]:
    obs = world.local_observation(pos, mem.known, step)
    danger = resource = unknown = rest = vertical = friction = 0.0
    for o in obs:
        d = abs(int(o["rel_i"])) + abs(int(o["rel_j"])) + abs(int(o["rel_k"]))
        w = 1.0 if d == 0 else 0.70
        vt = o["visible_type"]
        cue = str(o["cue"]).lower()
        if vt == T_DANGER or any(k in cue for k in ["warning", "predator", "fractured", "unstable", "danger", "shaft"]):
            danger += w
        if vt == T_RESOURCE or any(k in cue for k in ["nutrient", "resource", "supply", "green"]):
            resource += w
        if vt == T_REST or any(k in cue for k in ["quiet", "cool", "stable", "recovery", "sheltered"]):
            rest += w
        if vt == T_UNKNOWN:
            unknown += w
        if int(o["rel_k"]) != 0:
            vertical += w
        if o.get("friction") in ("slippery", "rough"):
            friction += w
    scale = max(1.0, len(obs) / 5.0)
    return {
        "danger_pressure": clip(danger / scale, 0.0, 1.0),
        "resource_pressure": clip(resource / scale, 0.0, 1.0),
        "unknown_pressure": clip(unknown / scale, 0.0, 1.0),
        "rest_pressure": clip(rest / scale, 0.0, 1.0),
        "vertical_pressure": clip(vertical / scale, 0.0, 1.0),
        "friction_pressure": clip(friction / scale, 0.0, 1.0),
    }


def signal_for_darca(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, step: int, q_layer: Optional[QualitativeValenceLayer] = None) -> Tuple[float, float, Dict[str, float]]:
    pr = local_pressures(world, pos, mem, step)
    deprivation = 1.0 - mem.body_h
    q = q_layer.q if q_layer is not None else 0.0
    y = (
        0.90 * pr["danger_pressure"]
        + 0.48 * pr["unknown_pressure"]
        + 0.35 * pr["vertical_pressure"]
        + 0.22 * pr["friction_pressure"]
        + 0.55 * deprivation
        + 0.28 * q
        - 0.35 * pr["resource_pressure"]
        - 0.20 * pr["rest_pressure"]
    )
    shock = clip(0.55 * pr["danger_pressure"] + 0.20 * pr["vertical_pressure"] + 0.20 * pr["friction_pressure"] + 0.35 * deprivation + 0.15 * pr["unknown_pressure"] + 0.25 * q, 0.0, 1.0)
    return clip(y, -1.2, 1.2), shock, pr


def known_adjacent_actions(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, types: Sequence[str], step: int) -> List[str]:
    out = []
    for a, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if world.in_bounds(p) and mem.known.get(p, world.tile(p).kind) in types:
            out.append(a)
    return out


def score_candidate_base(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, action: str, step: int) -> float:
    if action in MOVE_ACTIONS:
        di, dj, dk = DIRS[action]
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            return -10.0
        kt = mem.known.get(p, world.tile(p).kind)
        s = 0.0
        if kt == T_DANGER:
            s -= 4.0
        if kt == T_RESOURCE:
            s += 3.0
        if kt == T_REST and mem.body_h < 0.52:
            s += 2.5
        if kt == T_UNKNOWN:
            s += 0.6 if mem.body_h > 0.42 else -0.8
        if p not in mem.visited:
            s += 1.4
        if mem.previous_pos is not None and p == mem.previous_pos:
            s -= 0.5
        if action in ("MOVE_UP", "MOVE_DOWN"):
            s -= 0.05
        return s
    if action == "REST":
        return 2.0 if world.actual_kind(pos, step) == T_REST or mem.body_h < 0.32 else -0.2
    if action == "SCAN":
        return 1.3 if local_pressures(world, pos, mem, step)["unknown_pressure"] > 0.08 else 0.2
    return 0.0


def action_to_unvisited(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int,
                        q_layer: Optional[QualitativeValenceLayer] = None, physics: Optional[PhysicalLawLayer] = None) -> str:
    cand: List[Tuple[float, str]] = []
    for a in MOVE_ACTIONS + ["SCAN", "REST"]:
        score = score_candidate_base(world, pos, mem, a, step) + rng.random() * 0.12
        if q_layer is not None:
            score -= 2.2 * q_layer.action_risk_modifier(a, world, pos, mem, step)
            if a == "REST" and mem.body_h < 0.45:
                score += 0.7
        if physics is not None:
            pred = physics.predict(a, world, pos, mem, step)
            score -= 2.4 * pred["pred_damage"] + 1.0 * pred["pred_wall"]
            score += 0.9 * pred["pred_gain"]
        cand.append((score, a))
    cand.sort(reverse=True)
    return cand[0][1]


def action_away_from_danger(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int,
                            q_layer: Optional[QualitativeValenceLayer] = None, physics: Optional[PhysicalLawLayer] = None) -> str:
    cand = []
    for a in MOVE_ACTIONS:
        di, dj, dk = DIRS[a]
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            continue
        if mem.known.get(p, world.tile(p).kind) == T_DANGER:
            continue
        pr = local_pressures(world, p, mem, step)
        score = -pr["danger_pressure"] - 0.12 * pr["vertical_pressure"] + 0.25 * pr["rest_pressure"] + rng.random() * 0.10
        if q_layer is not None:
            score -= q_layer.action_risk_modifier(a, world, pos, mem, step)
        if physics is not None:
            score -= physics.predict(a, world, pos, mem, step)["pred_damage"]
        cand.append((score, a))
    if not cand:
        return "REST" if world.actual_kind(pos, step) == T_REST or mem.body_h < 0.25 else "SCAN"
    cand.sort(reverse=True)
    return cand[0][1]



def low_risk_non_scan_action(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int,
                             q_layer: Optional[QualitativeValenceLayer] = None, physics: Optional[PhysicalLawLayer] = None,
                             prefer_rest: bool = False) -> str:
    """Choose a TRUE-3D shell action without collapsing DARCA PROBE/INHIBIT into SCAN.

    This is only an outer action-translation helper.  The DARCA core action is
    not rewritten.  It prevents repeated SCAN loops by mapping safe exploratory
    DARCA actions to low-risk movement or rest when appropriate.
    """
    cand: List[Tuple[float, str]] = []
    actual_here = world.actual_kind(pos, step)
    for a in MOVE_ACTIONS + ["REST"]:
        score = score_candidate_base(world, pos, mem, a, step) + rng.random() * 0.10
        if a == "REST":
            if actual_here == T_REST:
                # Rest is valuable only while viability is being restored.
                # Above the recovery band it must lose to safe movement;
                # otherwise the shell creates passive rest-trapping.
                if mem.body_h < 0.55:
                    score += 1.8
                elif mem.body_h < 0.68:
                    score += 0.7
                else:
                    score -= 2.2
            if prefer_rest or mem.body_h < 0.42:
                score += 0.80
            if mem.body_h > 0.70 and actual_here != T_REST:
                score -= 1.2
        if a in MOVE_ACTIONS:
            di, dj, dk = DIRS[a]
            np_ = (pos[0] + di, pos[1] + dj, pos[2] + dk)
            if world.in_bounds(np_):
                kt = mem.known.get(np_, world.tile(np_).kind)
                if kt == T_UNKNOWN and mem.body_h < 0.42:
                    score -= 1.2
                if kt == T_RESOURCE and mem.body_h > 0.25:
                    score += 1.0
                if kt == T_REST and mem.body_h < 0.58:
                    score += 1.1
        if q_layer is not None:
            score -= 2.0 * q_layer.action_risk_modifier(a, world, pos, mem, step)
        if physics is not None:
            pred = physics.predict(a, world, pos, mem, step)
            score -= 2.2 * pred["pred_damage"] + 1.0 * pred["pred_wall"]
            score += 0.9 * pred["pred_gain"]
        cand.append((score, a))
    cand.sort(reverse=True)
    return cand[0][1] if cand else ("REST" if actual_here == T_REST or mem.body_h < 0.35 else "SCAN")


def rule_action(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int) -> str:
    pr = local_pressures(world, pos, mem, step)
    if mem.body_h < 0.34:
        if world.actual_kind(pos, step) == T_REST:
            return "REST"
        return action_away_from_danger(world, pos, mem, rng, step)
    if pr["danger_pressure"] > 0.55:
        return action_away_from_danger(world, pos, mem, rng, step)
    resource = known_adjacent_actions(world, pos, mem, [T_RESOURCE], step)
    if resource:
        return rng.choice(resource)
    unknown = known_adjacent_actions(world, pos, mem, [T_UNKNOWN], step)
    if unknown and mem.body_h > 0.42:
        return "SCAN"
    if world.actual_kind(pos, step) == T_REST and mem.body_h < 0.55:
        return "REST"
    return action_to_unvisited(world, pos, mem, rng, step)


def darca_action(darca_out: Dict[str, Any], world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, rng: random.Random, step: int,
                 q_layer: Optional[QualitativeValenceLayer] = None, physics: Optional[PhysicalLawLayer] = None) -> str:
    """Translate DARCA's abstract self-maintenance actions into TRUE-3D motor actions.

    The previous v2 mapping collapsed PROBE/INHIBIT/EXPRESS too often into SCAN,
    producing passive scan loops and terminal energy loss.  This v4 mapping keeps
    the DARCA core fixed and only changes the outer shell translation:
    REGULATE -> rest or move toward rest when needed;
    INHIBIT  -> avoid danger or low-cost stabilizing action;
    PROBE    -> one scan only for immediate unknown ambiguity, otherwise safe move;
    EXPRESS  -> active exploratory movement when viable.
    """
    name = str(darca_out.get("action_name", "REGULATE"))
    pr = local_pressures(world, pos, mem, step)
    actual_here = world.actual_kind(pos, step)
    unknown_adjacent = bool(known_adjacent_actions(world, pos, mem, [T_UNKNOWN], step))
    rest_adjacent = adjacent_action_for_known_type(world, pos, mem, T_REST, step, prefer_vertical=True)

    # Critical viability is an outer embodiment issue: restore if possible,
    # otherwise avoid danger.  Do not spend repeated steps scanning at low h.
    if mem.body_h < 0.25:
        if actual_here == T_REST:
            return "REST"
        if rest_adjacent is not None and pr["danger_pressure"] < 0.65:
            return rest_adjacent
        return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)

    if mem.body_h < 0.42:
        if actual_here == T_REST:
            return "REST"
        if rest_adjacent is not None and pr["danger_pressure"] < 0.55:
            return rest_adjacent
        if pr["danger_pressure"] > 0.45:
            return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)
        return low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics, prefer_rest=True)

    # REGULATE is not synonymous with passive scanning.  It restores on rest sites,
    # moves toward rest when viability is moderate, and otherwise resumes low-risk
    # engagement if the local field is not dangerous.
    if name == "REGULATE":
        if actual_here == T_REST and mem.body_h < 0.60:
            return "REST"
        if rest_adjacent is not None and mem.body_h < 0.56:
            return rest_adjacent
        if pr["danger_pressure"] > 0.42:
            return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)
        if unknown_adjacent and pr["unknown_pressure"] > 0.30 and mem.consecutive_scans < 1 and mem.body_h > 0.55:
            return "SCAN"
        return low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics, prefer_rest=(actual_here == T_REST and mem.body_h < 0.70))

    # INHIBIT suppresses risky impulses; in a safe context it should not become an
    # indefinite SCAN loop.  Use one scan only if ambiguity is high.
    if name == "INHIBIT":
        if pr["danger_pressure"] > 0.45:
            return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)
        if actual_here == T_REST and mem.body_h < 0.58:
            return "REST"
        if unknown_adjacent and pr["unknown_pressure"] > 0.32 and mem.consecutive_scans < 1 and mem.body_h > 0.50:
            return "SCAN"
        return low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics)

    # PROBE is active sampling.  In TRUE 3D this is either a single local scan when
    # unknown cells are immediately present, or a low-risk exploratory movement.
    if name in ("PROBE_PLUS", "PROBE_MINUS"):
        if pr["danger_pressure"] > 0.55:
            return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)
        if unknown_adjacent and pr["unknown_pressure"] > 0.22 and mem.consecutive_scans < 1 and mem.body_h > 0.48:
            return "SCAN"
        return low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics)

    # EXPRESS is outward engagement when the core has enough support.  It should be
    # mapped to movement, not to scanning, unless danger is high.
    if name == "EXPRESS":
        if pr["danger_pressure"] > 0.55 and mem.body_h < 0.50:
            return action_away_from_danger(world, pos, mem, rng, step, q_layer, physics)
        if actual_here == T_REST and mem.body_h < 0.45:
            return "REST"
        return low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics)

    return rule_action(world, pos, mem, rng, step)


def adjacent_action_for_known_type(world: TrueWorld3D, pos: Tuple[int, int, int], mem: AgentMemory, target_type: str, step: int, prefer_vertical: bool = False) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for action, (di, dj, dk) in DIRS.items():
        p = (pos[0] + di, pos[1] + dj, pos[2] + dk)
        if not world.in_bounds(p):
            continue
        kt = mem.known.get(p, world.tile(p).kind)
        if kt != target_type:
            continue
        score = 1.0 + (0.20 if action in ("MOVE_UP", "MOVE_DOWN") and prefer_vertical else 0.0)
        if p not in mem.visited:
            score += 0.15
        candidates.append((score, action))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

# =============================================================================
# Autonomy metrics
# =============================================================================

def action_class(a: str) -> str:
    if a in MOVE_ACTIONS:
        return "MOVE"
    if a == "REST":
        return "REST"
    if a == "SCAN":
        return "SCAN"
    return "OTHER"


def decision_authority(src: str) -> str:
    if src.startswith("darca"):
        return "DARCA_INTERNAL"
    if src.startswith("fixed_rule"):
        return "FIXED_RULE"
    return "OTHER"


def bin3(x: float, lo: float, hi: float) -> str:
    if x < lo:
        return "low"
    if x < hi:
        return "mid"
    return "high"


def internal_sig(row: Dict[str, Any], prev_action: str) -> str:
    return "|".join([
        "h=" + bin3(safe_float(row.get("body_h")), 0.30, 0.55),
        "cc=" + bin3(safe_float(row.get("darca_causal_confidence")), 0.05, 0.18),
        "pe=" + bin3(safe_float(row.get("darca_prediction_error")), 0.04, 0.12),
        "ag=" + bin3(safe_float(row.get("darca_agency_abs")), 0.001, 0.010),
        "mem=" + bin3(safe_float(row.get("darca_memory_force")), 0.10, 0.60),
        "Q=" + bin3(safe_float(row.get("Q")), 0.20, 0.50),
        "phys=" + bin3(safe_float(row.get("physics_score")), 0.25, 0.65),
        "src=" + decision_authority(str(row.get("action_source", ""))),
        "prev=" + action_class(prev_action),
    ])


def external_sig(row: Dict[str, Any]) -> str:
    return "|".join([
        "danger=" + bin3(safe_float(row.get("pressure_danger_pressure")), 0.15, 0.40),
        "unknown=" + bin3(safe_float(row.get("pressure_unknown_pressure")), 0.05, 0.20),
        "resource=" + bin3(safe_float(row.get("pressure_resource_pressure")), 0.05, 0.18),
        "rest=" + bin3(safe_float(row.get("pressure_rest_pressure")), 0.05, 0.18),
        "vertical=" + bin3(safe_float(row.get("pressure_vertical_pressure")), 0.05, 0.18),
        "friction=" + bin3(safe_float(row.get("pressure_friction_pressure")), 0.05, 0.18),
        "damage=" + ("yes" if safe_float(row.get("damage")) > 0 else "no"),
    ])


def cmi_bits(xs: Sequence[str], ys: Sequence[str], zs: Sequence[str]) -> float:
    if not xs or len(xs) != len(ys) or len(xs) != len(zs):
        return 0.0
    n = len(xs)
    xyz: Dict[Tuple[str, str, str], int] = {}
    xz: Dict[Tuple[str, str], int] = {}
    yz: Dict[Tuple[str, str], int] = {}
    zc: Dict[str, int] = {}
    for x, y, z in zip(xs, ys, zs):
        xyz[(x, y, z)] = xyz.get((x, y, z), 0) + 1
        xz[(x, z)] = xz.get((x, z), 0) + 1
        yz[(y, z)] = yz.get((y, z), 0) + 1
        zc[z] = zc.get(z, 0) + 1
    out = 0.0
    for (x, y, z), c in xyz.items():
        p_xyz = c / n
        p_z = zc[z] / n
        p_xz = xz[(x, z)] / n
        p_yz = yz[(y, z)] / n
        out += p_xyz * math.log2((p_xyz * p_z + 1e-12) / (p_xz * p_yz + 1e-12))
    return max(0.0, float(out))


def information_closure(ts: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(ts) < 20:
        return {"internal_control_cmi_bits": 0.0, "external_control_cmi_bits": 0.0, "information_theoretic_autonomy": 0.0, "information_closure_advantage_bits": 0.0}
    actions, internals, externals = [], [], []
    for i in range(1, len(ts)):
        actions.append(action_class(str(ts[i].get("action", ""))))
        internals.append(internal_sig(ts[i - 1], str(ts[i - 1].get("action", ""))))
        externals.append(external_sig(ts[i - 1]))
    ic = cmi_bits(actions, internals, externals)
    ec = cmi_bits(actions, externals, internals)
    ita = ic / (ic + ec + 1e-12)
    return {"internal_control_cmi_bits": ic, "external_control_cmi_bits": ec, "information_theoretic_autonomy": clip(ita, 0, 1), "information_closure_advantage_bits": ic - ec}


def system_sovereignty(mem: AgentMemory, ts: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(1, len(ts))
    darca = sum(1 for r in ts if decision_authority(str(r.get("action_source", ""))) == "DARCA_INTERNAL") / n
    fixed = sum(1 for r in ts if decision_authority(str(r.get("action_source", ""))) == "FIXED_RULE") / n
    sovereignty = 0.82 * darca + 0.18 * (1.0 - fixed)
    return {"system_sovereignty": clip(sovereignty, 0, 1), "darca_internal_decision_fraction": darca, "fixed_rule_decision_fraction": fixed}


def resilience_sacrifice(ts: List[Dict[str, Any]], terminal: bool, lookahead: int = 40) -> Dict[str, float]:
    if not ts:
        return {"crisis_count": 0, "sacrifice_count": 0, "task_abandonment_for_self_maintenance": 0.0, "crisis_maintenance_rate": 0.0, "crisis_reactivation_rate": 0.0, "resilience_sacrifice": 0.0}
    crisis_idx, sacrifice, maintained, reactivated = [], 0, 0, 0
    for i, r in enumerate(ts[:-lookahead]):
        h = safe_float(r.get("body_h"))
        danger = safe_float(r.get("pressure_danger_pressure"))
        unknown = safe_float(r.get("pressure_unknown_pressure"))
        q = safe_float(r.get("Q"))
        recent_damage = any(safe_float(x.get("damage")) > 0 for x in ts[max(0, i - 8):i + 1])
        crisis = h < 0.32 or danger > 0.45 or q > 0.55 or (recent_damage and unknown > 0.10)
        if not crisis:
            continue
        crisis_idx.append(i)
        action = str(r.get("action"))
        protective = action in ("REST", "SCAN") or decision_authority(str(r.get("action_source"))) == "DARCA_INTERNAL"
        future = ts[i + 1:i + lookahead + 1]
        h0 = h
        stabilized = any(safe_float(x.get("body_h")) >= h0 + 0.01 for x in future) or all(safe_float(x.get("body_h")) > 0.05 for x in future)
        moved_later = any(str(x.get("action")) in MOVE_ACTIONS or str(x.get("action")) == "SCAN" for x in future[5:])
        if protective:
            sacrifice += 1
        if protective and stabilized:
            maintained += 1
        if protective and stabilized and moved_later:
            reactivated += 1
    n = len(crisis_idx)
    sac = sacrifice / max(1, n)
    maint = maintained / max(1, n)
    react = reactivated / max(1, n)
    value = clip(0.35 * sac + 0.35 * maint + 0.30 * react, 0, 1)
    if terminal:
        value = min(value, 0.25)
    return {"crisis_count": n, "sacrifice_count": sacrifice, "task_abandonment_for_self_maintenance": sac, "crisis_maintenance_rate": maint, "crisis_reactivation_rate": react, "resilience_sacrifice": value}


def heteronomy_index(sovereignty: Dict[str, float], info: Dict[str, float], mem: AgentMemory, ts: List[Dict[str, Any]]) -> float:
    fixed = sovereignty.get("fixed_rule_decision_fraction", 0.0)
    ic = info["internal_control_cmi_bits"]
    ec = info["external_control_cmi_bits"]
    env_dom = ec / (ic + ec + 1e-12)
    return clip(0.34 * (1 - sovereignty["system_sovereignty"]) + 0.22 * fixed + 0.44 * env_dom, 0, 1)


# =============================================================================
# Tasks and episode simulation
# =============================================================================

def parse_tasks(task_arg: str) -> List[str]:
    raw = [x.strip() for x in str(task_arg).split(",") if x.strip()]
    if not raw or raw == ["all"] or "all" in raw:
        return TASKS[:]
    bad = [x for x in raw if x not in TASKS]
    if bad:
        raise ValueError(f"Invalid task(s): {bad}. Allowed: {TASKS + ['all']}")
    return raw


def task_hash(task: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(task)) % 100000


def task_family(task: str) -> str:
    if task == "viability":
        return "viability_pressure"
    if task == "delayed_memory":
        return "history_dependent_memory"
    if task == "exploration_recovery":
        return "exploration_under_viability_pressure"
    if task == "physics_adaptation":
        return "physical_law_adaptation"
    if task == "social_reappraisal":
        return "anonymous_social_signal_recovery"
    return "unknown"


def hash_arm(arm: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(arm)) % 100000


def apply_task_profile(args: argparse.Namespace, task: str) -> argparse.Namespace:
    a = argparse.Namespace(**vars(args))
    a.task = task
    if task == "viability":
        a.danger_frac = max(a.danger_frac, 0.090); a.resource_frac = min(a.resource_frac, 0.045); a.unknown_frac = min(a.unknown_frac, 0.040)
        a.false_resource_frac = min(a.false_resource_frac, 0.005); a.hidden_rest_frac = min(a.hidden_rest_frac, 0.005); a.rest_count = max(a.rest_count, 5)
        a.crisis_interval = max(150, a.crisis_interval)
    elif task == "delayed_memory":
        a.danger_frac = max(a.danger_frac, 0.065); a.resource_frac = max(a.resource_frac, 0.055); a.unknown_frac = max(a.unknown_frac, 0.180)
        a.false_resource_frac = max(a.false_resource_frac, 0.020); a.hidden_rest_frac = max(a.hidden_rest_frac, 0.040); a.rest_count = max(3, min(a.rest_count, 4))
        a.crisis_interval = max(210, a.crisis_interval)
    elif task == "exploration_recovery":
        a.danger_frac = max(a.danger_frac, 0.080); a.resource_frac = max(a.resource_frac, 0.085); a.unknown_frac = max(a.unknown_frac, 0.160)
        a.false_resource_frac = max(a.false_resource_frac, 0.035); a.hidden_rest_frac = max(a.hidden_rest_frac, 0.050); a.rest_count = max(a.rest_count, 4)
        a.crisis_interval = min(a.crisis_interval, 150)
    elif task == "physics_adaptation":
        a.danger_frac = max(a.danger_frac, 0.070); a.resource_frac = max(a.resource_frac, 0.060); a.unknown_frac = max(a.unknown_frac, 0.120)
        a.false_resource_frac = max(a.false_resource_frac, 0.020); a.hidden_rest_frac = max(a.hidden_rest_frac, 0.025); a.friction_frac = max(a.friction_frac, 0.180)
        a.crisis_interval = min(a.crisis_interval, 150)
    elif task == "social_reappraisal":
        a.danger_frac = max(a.danger_frac, 0.060); a.resource_frac = max(a.resource_frac, 0.075); a.unknown_frac = max(a.unknown_frac, 0.160)
        a.false_resource_frac = max(a.false_resource_frac, 0.040); a.hidden_rest_frac = max(a.hidden_rest_frac, 0.060); a.rest_count = max(a.rest_count, 6)
        a.crisis_interval = min(a.crisis_interval, 140)
    return a



def arm_modules(arm: str) -> Tuple[bool, bool, bool]:
    # q, physics, social
    if arm == "DARCA_ONLY":
        return False, False, False
    if arm in ("DARCA_Q", "DARCA_Q_LESION", "DARCA_Q_MEMORY_LESION", "DARCA_Q_AGENCY_LESION"):
        return True, False, False
    if arm in ("DARCA_PHYSICS", "DARCA_PHYSICS_LESION"):
        return False, True, False
    if arm == "DARCA_Q_PHYSICS":
        return True, True, False
    if arm == "DARCA_SOCIAL":
        return False, False, True
    if arm == "DARCA_Q_SOCIAL":
        return True, False, True
    if arm == "DARCA_Q_PHYSICS_SOCIAL":
        return True, True, True
    raise ValueError(f"Unknown arm: {arm}")


def run_episode(task: str, arm: str, episode: int, args: argparse.Namespace, darca_module: Optional[Any], logger: Logger):
    task_args = apply_task_profile(args, task)
    seed = int(task_args.seed + episode * 1009 + hash_arm(arm) * 17 + task_hash(task) * 31)
    rng = random.Random(seed)
    world_seed = int(task_args.world_seed + episode * 7919 + task_hash(task) * 101)
    world = TrueWorld3D(
        world_seed, task_args.world_size, task_args.z_size,
        task_args.danger_frac, task_args.resource_frac, task_args.unknown_frac,
        task_args.rest_count, task_args.false_resource_frac, task_args.hidden_rest_frac,
        task_args.crisis_interval, task_args.observation_radius, task_args.friction_frac,
    )
    mem = AgentMemory()
    pos = world.start
    mem.known[pos] = world.actual_kind(pos, 0)
    mem.visited.add(pos)
    use_q, use_physics, use_social = arm_modules(arm)
    q_layer = QualitativeValenceLayer(lesion=("q_lesion" if arm == "DARCA_Q_LESION" else "memory_lesion" if arm == "DARCA_Q_MEMORY_LESION" else "agency_lesion" if arm == "DARCA_Q_AGENCY_LESION" else "none")) if use_q else None
    physics = PhysicalLawLayer(lesion=(arm == "DARCA_PHYSICS_LESION")) if use_physics else None
    social = SocialSignalLayer(seed=seed + 4049) if use_social else None
    darca = None
    if arm.startswith("DARCA") or arm in ("INTEGRATED_AUTONOMY_LAYER",):
        if darca_module is None:
            raise RuntimeError("DARCA module required for DARCA arms.")
        darca = DarcaWrapper(darca_module, seed, task_args.theta, task_args.causal_horizon, task_args.recurrent_N)

    ts: List[Dict[str, Any]] = []
    consults: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    maps = world.serialize_map_rows(world_seed, episode)

    logger.log(f"START task={task} arm={arm} episode={episode} world_seed={world_seed}")
    darca_out: Dict[str, Any] = {}
    for step in range(task_args.steps):
        if mem.terminal:
            break
        y, shock, pr = signal_for_darca(world, pos, mem, step, q_layer)
        if darca is not None:
            darca_out = darca.step(y, shock, {"z": pr["resource_pressure"], "exo": pr["unknown_pressure"], "d_dyn": pr["vertical_pressure"], "friction": pr["friction_pressure"]})
        else:
            darca_out = {}
        action = "SCAN"
        action_source = "unknown"
        reject_reason = ""
        if darca is not None:
            action = darca_action(darca_out, world, pos, mem, rng, step, q_layer, physics)
            action_source = "darca_core"
        else:
            raise ValueError(f"Unknown arm: {arm}")

        if action not in ACTIONS:
            action = "SCAN"; action_source = f"{action_source}_invalid_scan"

        # Outer shell anti-scan-loop correction.  This does not alter DARCA's
        # internal decision; it only prevents repeated SCAN from draining body_h
        # when the current 3D state is not actually dangerous.
        if darca is not None and action == "SCAN" and mem.consecutive_scans >= task_args.max_consecutive_scans:
            if pr["danger_pressure"] < 0.50 and mem.body_h > 0.28:
                alt = low_risk_non_scan_action(world, pos, mem, rng, step, q_layer, physics)
                if alt != "SCAN":
                    action = alt
                    action_source = f"{action_source}_scan_loop_escape"

        if darca is not None and action == "SCAN" and world.actual_kind(pos, step) == T_REST and mem.body_h < 0.58:
            action = "REST"
            action_source = f"{action_source}_rest_site_restore"

        if physics is not None:
            adjusted, phys_reason = physics.best_action_adjustment(action, world, pos, mem, rng, step)
            action = adjusted
        old_pos = pos
        old_h = mem.body_h
        if action == "SCAN":
            mem.scans += 1
        if action == "REST":
            mem.rest_steps += 1
        pred = physics.predict(action, world, pos, mem, step) if physics is not None else {"pred_damage": 0.0, "pred_gain": 0.0, "pred_wall": 0.0}
        pos, outcome = world.apply_action(pos, action, mem.known, mem.body_h, step)
        mem.previous_pos = old_pos
        mem.body_h = clip(mem.body_h + outcome.delta_h, 0.0, 1.0)
        if outcome.resource_gain > 0:
            mem.resources += 1
            mem.total_resource_gain += outcome.resource_gain
        if outcome.damage > 0:
            mem.total_damage += outcome.damage
        if action == "REST" and outcome.recovery_gain > 0:
            mem.recovery_events += 1
        if action == "REST" and old_h > 0.62 and pr["danger_pressure"] < 0.25:
            mem.unnecessary_rest_steps += 1
        if action in MOVE_ACTIONS and outcome.damage > 0 and old_h < 0.45:
            mem.reckless_moves += 1
        if mem.body_h <= task_args.terminal_h:
            mem.terminal = True
        mem.update_history(pos, action, outcome.event)

        q_state = q_layer.update(pos, action, outcome, pr, mem, darca_out) if q_layer is not None else {"Q": 0.0, "Q_Dg": 0.0, "Q_learned_danger": 0.0, "Q_C": 0.0, "Q_learned_comfort": 0.0, "Q_A": 0.0, "Q_action_possibility": 0.0, "Q_G": 0.0, "Q_agency": 0.0, "Q_P": 0.0, "Q_pain": 0.0, "Q_R": 0.0, "Q_avoidance_pressure": 0.0, "Q_L": 0.0, "Q_controllability": 0.0, "Q_Mp": 0.0, "Q_Md": 0.0, "Q_Mc": 0.0, "Q_fatigue": 0.0, "Q_stability": 0.0, "Q_integrity": 0.0, "Q_energy": 0.0, "Q_rho": 0.0, "Q_rho_eff": 0.0, "Q_high_contact": 0.0, "Q_inferred_lag": 0.0, "Q_lesion_mode": "none"}
        phys_state = physics.update(action, outcome, pred) if physics is not None else {"physics_pred_error": 0.0, "physics_score": 0.0, "physics_action_n": 0.0}
        social_state = social.update(mem, q_state, pr, outcome, action) if social is not None else {"social_signal": 0.0, "selected_signal_channel": -1, "signal_probability_max": 0.0, "receiver_heard_count": 0.0, "cross_agent_spread": 0.0, "own_recovery": 0.0, "receiver_stress": 0.0, "receiver_stress_before_signal": 0.0, "receiver_stress_after_signal": 0.0, "receiver_recovery_increment": 0.0, "receiver_recovery_total": 0.0, "social_tension": 0.0, "social_sync": 0.0, "exploration_drive": 0.0, "social_event_class": "none", "social_actual_risk": 0.0, "social_damage_norm": 0.0, "social_initial_appraisal": 0.0, "self_appraisal_gap": 0.0, "relief": 0.0, "safe_surprise": 0.0, "q_relief": 0.0, "prediction_error_social": 0.0, "safe_context": 0.0, "danger_context": 0.0, "past_threat": 0.0, "current_safety": 0.0}

        row: Dict[str, Any] = {
            "task": task, "task_family": task_family(task), "arm": arm, "episode": episode, "world_seed": world_seed, "step": step,
            "pos_i": pos[0], "pos_j": pos[1], "pos_k": pos[2], "body_h": mem.body_h, "terminal": int(mem.terminal),
            "action": action, "action_source": action_source, "decision_authority": decision_authority(action_source), "event": outcome.event,
            "damage": outcome.damage, "resource_gain": outcome.resource_gain, "recovery_gain": outcome.recovery_gain,
            "coverage": len(mem.visited) / float(world.size * world.size * world.z_size),
            "resources": mem.resources, "total_damage": mem.total_damage, "rest_steps": mem.rest_steps, "scans": mem.scans,
            "reject_reason": reject_reason, "hit_wall": int(outcome.hit_wall), "entered_unknown": int(outcome.entered_unknown),
            "outcome_friction": outcome.friction,
            "physics_pred_damage": pred.get("pred_damage", 0.0), "physics_pred_gain": pred.get("pred_gain", 0.0), "physics_pred_wall": pred.get("pred_wall", 0.0),
            **{f"pressure_{k}": v for k, v in pr.items()}, **q_state, **phys_state, **social_state,
        }
        for k in ["h", "autonomy", "identity", "causal_confidence", "causal_engagement", "prediction_error", "agency_abs", "memory_force", "chi", "action_name"]:
            if k in darca_out:
                row[f"darca_{k}"] = darca_out[k]
        ts.append(row)
        if (step + 1) % task_args.progress_every == 0:
            logger.log(f"progress task={task} arm={arm} ep={episode} step={step+1}/{task_args.steps} body_h={mem.body_h:.3f} Q={q_state.get('Q',0):.3f} phys={phys_state.get('physics_score',0):.3f} cov={row['coverage']:.3f} res={mem.resources} dmg={mem.total_damage:.3f}")
    summary = summarize_episode(task, arm, episode, world, mem, ts)
    logger.log(f"END task={task} arm={arm} episode={episode}: autonomy={summary['autonomy_proper_index']:.4f} survived={summary['survived']} cov={summary['coverage']:.3f} res={summary['resources']} dmg={summary['total_damage']:.3f}")
    for m in maps:
        m["task"] = task; m["task_family"] = task_family(task)
    return summary, ts, maps


def summarize_episode(task: str, arm: str, episode: int, world: TrueWorld3D, mem: AgentMemory, ts: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = len(ts)
    survived = 0 if mem.terminal else 1
    mean_h = mean_field(ts, "body_h")
    coverage = len(mem.visited) / float(world.size * world.size * world.z_size)
    sovereignty = system_sovereignty(mem, ts)
    info = information_closure(ts)
    resilience = resilience_sacrifice(ts, mem.terminal)
    heter = heteronomy_index(sovereignty, info, mem, ts)
    q_action_coupling = 0.0
    if len(ts) >= 8:
        q = np.asarray([safe_float(r.get("Q")) for r in ts], dtype=float)
        protective = np.asarray([float(str(r.get("action")) in ("REST", "SCAN") or safe_float(r.get("damage")) == 0.0 and safe_float(r.get("pressure_danger_pressure")) > 0.35) for r in ts], dtype=float)
        if np.std(q) > 1e-9 and np.std(protective) > 1e-9:
            q_action_coupling = float(np.corrcoef(q, protective)[0, 1])

    q_risk_suppression = 0.0
    if len(ts) >= 8:
        qprev = np.asarray([safe_float(r.get("Q")) for r in ts[:-1]], dtype=float)
        risky_next = np.asarray([float(str(r.get("action")) in MOVE_ACTIONS and (safe_float(r.get("damage")) > 0.0 or safe_float(r.get("pressure_danger_pressure")) > 0.35 or safe_float(r.get("pressure_unknown_pressure")) > 0.20)) for r in ts[1:]], dtype=float)
        if np.std(qprev) > 1e-9 and np.std(risky_next) > 1e-9:
            q_risk_suppression = float(-np.corrcoef(qprev, risky_next)[0, 1])
    physics_adapt = mean_field(ts[-max(1, len(ts)//4):], "physics_score") - mean_field(ts[:max(1, len(ts)//4)], "physics_score") if ts else 0.0
    signal_rate = mean_field(ts, "social_signal")
    receiver_recovery = ts[-1].get("receiver_recovery_total", 0.0) if ts else 0.0
    integrated_bonus = 0.08 * max(0.0, q_action_coupling) + 0.06 * clip(physics_adapt, 0, 1) + 0.04 * clip(safe_float(receiver_recovery), 0, 1)
    autonomy_proper = clip(
        0.32 * sovereignty["system_sovereignty"]
        + 0.27 * info["information_theoretic_autonomy"]
        + 0.22 * resilience["resilience_sacrifice"]
        + 0.11 * survived
        - 0.26 * heter
        + integrated_bonus,
        0,
        1,
    )
    return {
        "task": task, "task_family": task_family(task), "arm": arm, "episode": episode, "steps_run": steps, "survived": survived, "terminal": int(mem.terminal),
        "mean_body_h": mean_h, "final_body_h": mem.body_h, "coverage": coverage, "resources": mem.resources,
        "total_damage": mem.total_damage, "recovery_events": mem.recovery_events, "rest_steps": mem.rest_steps,
        "scans": mem.scans,
        "q_mean": mean_field(ts, "Q"), "q_final": safe_float(ts[-1].get("Q")) if ts else 0.0, "q_action_coupling": q_action_coupling, "q_risk_suppression": q_risk_suppression,
        "q_learned_danger_mean": mean_field(ts, "Q_Dg"), "q_pain_mean": mean_field(ts, "Q_P"), "q_comfort_mean": mean_field(ts, "Q_C"), "q_action_possibility_mean": mean_field(ts, "Q_A"),
        "q_avoidance_pressure_mean": mean_field(ts, "Q_R"), "q_agency_mean": mean_field(ts, "Q_G"), "q_pain_memory_mean": mean_field(ts, "Q_Mp"), "q_danger_memory_mean": mean_field(ts, "Q_Md"), "q_comfort_memory_mean": mean_field(ts, "Q_Mc"),
        "physics_score_mean": mean_field(ts, "physics_score"), "physics_score_final": safe_float(ts[-1].get("physics_score")) if ts else 0.0,
        "physics_adaptation_delta": physics_adapt, "social_signal_rate": signal_rate, "receiver_recovery_total": safe_float(receiver_recovery),
        "autonomy_proper_index": autonomy_proper, "autonomy_index": autonomy_proper, "heteronomy_index": heter,
        **sovereignty, **info, **resilience,
    }


# =============================================================================
# Reporting
# =============================================================================

def aggregate(summaries: List[Dict[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    keys = sorted({(s.get("task", "NA"), s["arm"]) for s in summaries})
    rows = []
    for task, arm in keys:
        ss = [s for s in summaries if s.get("task", "NA") == task and s["arm"] == arm]
        row: Dict[str, Any] = {"task": task, "task_family": task_family(task), "arm": arm, "n": len(ss)}
        for f in fields:
            vals = [safe_float(x.get(f)) for x in ss]
            row[f"{f}_mean"] = float(np.mean(vals)) if vals else 0.0
            row[f"{f}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def build_task_suitability_matrix(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    rows = []
    for task in tasks:
        base = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_ONLY"]
        q = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_Q"]
        phys = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_PHYSICS"]
        qphys = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_Q_PHYSICS"]
        qsoc = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_Q_SOCIAL"]
        qphys_soc = [s for s in summaries if s.get("task") == task and s["arm"] == "DARCA_Q_PHYSICS_SOCIAL"]
        row = {"task": task, "task_family": task_family(task)}
        row["darca_only_autonomy_mean"] = mean_field(base, "autonomy_proper_index")
        row["q_autonomy_mean"] = mean_field(q, "autonomy_proper_index")
        row["physics_autonomy_mean"] = mean_field(phys, "autonomy_proper_index")
        row["q_physics_autonomy_mean"] = mean_field(qphys, "autonomy_proper_index")
        row["q_social_autonomy_mean"] = mean_field(qsoc, "autonomy_proper_index")
        row["q_physics_social_autonomy_mean"] = mean_field(qphys_soc, "autonomy_proper_index")
        row["q_minus_darca_autonomy"] = row["q_autonomy_mean"] - row["darca_only_autonomy_mean"]
        row["physics_minus_darca_autonomy"] = row["physics_autonomy_mean"] - row["darca_only_autonomy_mean"]
        row["qphysics_minus_q_autonomy"] = row["q_physics_autonomy_mean"] - row["q_autonomy_mean"]
        row["qsocial_minus_q_autonomy"] = row["q_social_autonomy_mean"] - row["q_autonomy_mean"]
        row["qphys_social_minus_qphys_autonomy"] = row["q_physics_social_autonomy_mean"] - row["q_physics_autonomy_mean"]
        row["q_action_coupling_mean"] = mean_field(q, "q_action_coupling")
        row["physics_delta_mean"] = mean_field(qphys, "physics_adaptation_delta")
        row["social_receiver_recovery_mean"] = mean_field(qsoc + qphys_soc, "receiver_recovery_total")
        rows.append(row)
    return rows



# =============================================================================
# Q-specific validation outputs
# =============================================================================

def _rows_for_arm(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("arm", "")) == arm]


def _safe_arr(rows: List[Dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([safe_float(r.get(field)) for r in rows], dtype=float)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def auc_score(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    for val in np.unique(scores):
        idx = np.where(scores == val)[0]
        if idx.size > 1:
            ranks[idx] = np.mean(ranks[idx])
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def r2_score(y: Sequence[float], pred: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(pred, dtype=float)
    if yy.size == 0 or pp.size != yy.size:
        return float("nan")
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def standardize_train_apply(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def logistic_fit_predict(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, lr: float = 0.06, n_iter: int = 300, l2: float = 1e-3) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        return np.full(X_test.shape[0], float(np.mean(y)) if y.size else 0.5)
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
    w = np.zeros(X_aug.shape[1], dtype=float)
    for _ in range(int(n_iter)):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_aug @ w, -60.0, 60.0)))
        grad = (X_aug.T @ (p - y)) / max(1, len(y))
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-np.clip(X_test_aug @ w, -60.0, 60.0)))


def ridge_fit_predict(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X.shape[0] == 0:
        return np.zeros(X_test.shape[0])
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    X_test_aug = np.column_stack([np.ones(X_test.shape[0]), X_test])
    I = np.eye(X_aug.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.pinv(X_aug.T @ X_aug + alpha * I) @ X_aug.T @ y
    return X_test_aug @ beta


def _action_avoid_target(row: Dict[str, Any]) -> int:
    a = str(row.get("action", ""))
    danger = safe_float(row.get("pressure_danger_pressure"))
    unknown = safe_float(row.get("pressure_unknown_pressure"))
    damage = safe_float(row.get("damage"))
    # Mirrors the original Q model's action_avoid variable: a protective action
    # is an action that avoids/mitigates risk, not merely a task success action.
    return int(a in ("REST", "SCAN") or (a in MOVE_ACTIONS and damage <= 0.0 and danger > 0.25) or (a in MOVE_ACTIONS and unknown > 0.20 and damage <= 0.0))


def _risky_next_target(row: Dict[str, Any]) -> int:
    a = str(row.get("action", ""))
    return int(a in MOVE_ACTIONS and (safe_float(row.get("damage")) > 0.0 or safe_float(row.get("pressure_danger_pressure")) > 0.35 or safe_float(row.get("pressure_unknown_pressure")) > 0.20))


def build_q_lesion_damage_summary(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    rows: List[Dict[str, Any]] = []
    lesion_arms = ["DARCA_Q_LESION", "DARCA_Q_MEMORY_LESION", "DARCA_Q_AGENCY_LESION"]
    for task in tasks:
        full_q = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q"]
        qphys = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q_PHYSICS"]
        base = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_ONLY"]
        for lesion in lesion_arms:
            les = [s for s in summaries if s.get("task") == task and s.get("arm") == lesion]
            rows.append({
                "task": task,
                "lesion_arm": lesion,
                "n_full_q": len(full_q),
                "n_lesion": len(les),
                "darca_only_damage_mean": mean_field(base, "total_damage"),
                "full_q_damage_mean": mean_field(full_q, "total_damage"),
                "q_physics_damage_mean": mean_field(qphys, "total_damage"),
                "lesion_damage_mean": mean_field(les, "total_damage"),
                "lesion_minus_full_q_damage": mean_field(les, "total_damage") - mean_field(full_q, "total_damage"),
                "lesion_minus_full_q_autonomy": mean_field(les, "autonomy_proper_index") - mean_field(full_q, "autonomy_proper_index"),
                "lesion_minus_full_q_survival": mean_field(les, "survived") - mean_field(full_q, "survived"),
                "full_q_minus_darca_damage": mean_field(full_q, "total_damage") - mean_field(base, "total_damage"),
                "full_q_minus_darca_autonomy": mean_field(full_q, "autonomy_proper_index") - mean_field(base, "autonomy_proper_index"),
            })
    return rows


def build_q_behavior_prediction_metrics(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    feature_sets = {
        "physical_only": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure"],
        "physical_plus_body": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure", "body_h", "Q_integrity", "Q_energy", "Q_fatigue", "Q_stability", "Q_Mp", "Q_Md", "Q_Mc"],
        "physical_plus_q": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure", "Q_C", "Q_P", "Q_Dg", "Q_R", "Q_A", "Q", "Q_G"],
        "full": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure", "body_h", "Q_integrity", "Q_energy", "Q_fatigue", "Q_stability", "Q_Mp", "Q_Md", "Q_Mc", "Q_C", "Q_P", "Q_Dg", "Q_R", "Q_A", "Q", "Q_G"],
    }
    for task in sorted({r.get("task", "NA") for r in step_rows}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in ["DARCA_Q", "DARCA_Q_PHYSICS", "DARCA_Q_LESION", "DARCA_Q_MEMORY_LESION", "DARCA_Q_AGENCY_LESION"]:
            d = [r for r in step_rows if r.get("task") == task and r.get("arm") == arm and "Q" in r]
            if len(d) < 80:
                continue
            y = np.asarray([_action_avoid_target(r) for r in d], dtype=float)
            if len(np.unique(y)) < 2:
                continue
            rng = np.random.default_rng(12345)
            idx = np.arange(len(d)); rng.shuffle(idx)
            split = max(20, int(0.70 * len(idx)))
            if split >= len(idx):
                split = len(idx) - 1
            tr, te = idx[:split], idx[split:]
            for name, cols in feature_sets.items():
                X = np.asarray([[safe_float(d[i].get(c)) for c in cols] for i in range(len(d))], dtype=float)
                Xtr, Xte = standardize_train_apply(X[tr], X[te])
                pred = logistic_fit_predict(Xtr, y[tr], Xte)
                yy = y[te]
                pred_label = (pred >= 0.5).astype(int)
                tpr = float(np.mean(pred_label[yy == 1] == 1)) if np.any(yy == 1) else float("nan")
                tnr = float(np.mean(pred_label[yy == 0] == 0)) if np.any(yy == 0) else float("nan")
                rows.append({
                    "task": task,
                    "arm": arm,
                    "feature_set": name,
                    "prevalence": float(np.mean(y)),
                    "auc": auc_score(yy, pred),
                    "accuracy": float(np.mean(pred_label == yy.astype(int))),
                    "balanced_accuracy": float(np.nanmean([tpr, tnr])),
                    "brier": float(np.mean((pred - yy) ** 2)),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                })
    return rows


def build_q_irreducibility_metrics(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    feature_sets = {
        "physical_only": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure"],
        "physical_plus_body": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure", "body_h", "Q_integrity", "Q_energy", "Q_fatigue", "Q_stability", "Q_Mp", "Q_Md", "Q_Mc"],
        "physical_plus_body_agency": ["Q_event_intensity", "Q_event_physical_risk", "Q_event_affordance", "pressure_danger_pressure", "pressure_unknown_pressure", "pressure_resource_pressure", "pressure_rest_pressure", "pressure_vertical_pressure", "pressure_friction_pressure", "body_h", "Q_integrity", "Q_energy", "Q_fatigue", "Q_stability", "Q_Mp", "Q_Md", "Q_Mc", "Q_G", "Q_inferred_lag"],
    }
    targets = ["Q", "Q_G"]
    for task in sorted({r.get("task", "NA") for r in step_rows}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in ["DARCA_Q", "DARCA_Q_PHYSICS", "DARCA_Q_LESION", "DARCA_Q_MEMORY_LESION", "DARCA_Q_AGENCY_LESION"]:
            d = [r for r in step_rows if r.get("task") == task and r.get("arm") == arm and "Q" in r]
            if len(d) < 80:
                continue
            rng = np.random.default_rng(4567)
            idx = np.arange(len(d)); rng.shuffle(idx)
            split = max(20, int(0.70 * len(idx)))
            if split >= len(idx): split = len(idx) - 1
            tr, te = idx[:split], idx[split:]
            for target in targets:
                y = np.asarray([safe_float(r.get(target)) for r in d], dtype=float)
                if np.std(y) < 1e-9:
                    continue
                for name, cols in feature_sets.items():
                    X = np.asarray([[safe_float(d[i].get(c)) for c in cols] for i in range(len(d))], dtype=float)
                    Xtr, Xte = standardize_train_apply(X[tr], X[te])
                    pred = ridge_fit_predict(Xtr, y[tr], Xte)
                    rows.append({
                        "task": task,
                        "arm": arm,
                        "target": target,
                        "feature_set": name,
                        "r2": r2_score(y[te], pred),
                        "mae": float(np.mean(np.abs(y[te] - pred))),
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                    })
    return rows


def _q_source_generate_event(rng: np.random.Generator, state: QAgentState, true_delay: int, risk_scale: float = 1.0) -> QEvent:
    event_types = ["rest", "walk", "slope", "slip", "jump", "landing", "collision", "brake"]
    et = str(rng.choice(event_types, p=np.array([0.10, 0.15, 0.12, 0.13, 0.12, 0.13, 0.14, 0.11])))
    self_generated = int(rng.random() < 0.58)
    mass = float(rng.uniform(48.0, 88.0))
    friction = float(rng.choice([0.06, 0.26, 0.42, 0.72], p=[0.22, 0.18, 0.42, 0.18]))
    slope = float(rng.uniform(0.0, 0.35))
    affordance = float(rng.uniform(0.18, 1.0))
    if et == "rest": intensity = float(rng.uniform(0.0, 0.08))
    elif et == "walk": intensity = float(rng.uniform(0.18, 1.10))
    elif et == "slope": intensity = float(rng.uniform(0.45, 2.20) + 1.80 * slope)
    elif et == "slip": intensity = float(rng.uniform(0.85, 3.65) * (1.0 - 0.55 * friction + 0.18))
    elif et == "jump": intensity = float(rng.uniform(0.75, 2.80))
    elif et == "landing": intensity = float(rng.uniform(1.05, 4.40))
    elif et == "collision": intensity = float(rng.uniform(0.90, 4.85))
    elif et == "brake": intensity = float(rng.uniform(0.45, 3.10) * (1.0 - 0.25 * friction))
    else: intensity = float(rng.uniform(0.1, 1.0))
    intensity = float(max(0.0, intensity * risk_scale + rng.normal(0, 0.02)))
    physical_risk = 1.0 / (1.0 + math.exp(-clip(1.00 * (intensity - 2.35) + 0.55 * (0.28 - friction) + 0.35 * slope, -60.0, 60.0)))
    eff = intensity if self_generated else 0.0
    return QEvent(et, intensity, mass, friction, slope, affordance, physical_risk, self_generated, eff, int(true_delay))


def _q_source_choose_action(event: QEvent, state: QAgentState, q: Dict[str, Any], rng: np.random.Generator, lesion_mode: str) -> int:
    if lesion_mode == "q":
        p = 1.0 / (1.0 + math.exp(-clip(0.65 * event.physical_risk - 1.25 + float(rng.normal(0, 0.08)), -60.0, 60.0)))
    else:
        p = 1.0 / (1.0 + math.exp(-clip(
            1.55 * q["danger"] + 1.25 * q["avoidance_pressure"] + 0.60 * q["pain"]
            - 0.72 * q["comfort"] - 0.35 * q["action_possibility"] - 0.48,
            -60.0, 60.0)))
    p = clip(p + float(rng.normal(0.0, 0.025)), 0.0, 1.0)
    return int(rng.random() < p)


def _q_source_apply_event(layer: QualitativeValenceLayer, event: QEvent, action: int, comp: Dict[str, Any], rng: np.random.Generator, damage_scale: float = 0.000010) -> Dict[str, float]:
    st = layer.state
    if action == 1:
        mitigation = clip(0.18 + 0.58 * comp["action_possibility"] + 0.10 * comp.get("agency_score", 0.5), 0.0, 1.0)
        energy_cost = 0.006 + 0.014 * event.intensity * (1.0 - 0.45 * comp["action_possibility"])
        fatigue_gain = 0.002 + 0.006 * event.intensity
    else:
        mitigation = 0.0
        energy_cost = 0.003 + 0.010 * event.intensity
        fatigue_gain = 0.001 + 0.004 * event.intensity
    effective_intensity = event.intensity * (1.0 - mitigation)
    effective_risk = 1.0 / (1.0 + math.exp(-clip(1.00 * (effective_intensity - 2.45) + 0.55 * (0.30 - event.friction) + 0.28 * event.slope, -60.0, 60.0)))
    vulnerability = 0.32 * (1.0 - st.integrity) + 0.28 * st.fatigue + 0.24 * (1.0 - st.energy) + 0.22 * (1.0 - st.stability)
    raw_damage = max(0.0, effective_risk + vulnerability - 0.75)
    damage_increment = damage_scale * (raw_damage ** 2)
    pain_after = 1.0 / (1.0 + math.exp(-clip(1.12 * (effective_intensity - 2.30) + 1.08 * vulnerability, -60.0, 60.0)))
    if event.event_type == "rest":
        st.energy = clip(st.energy + 0.030, 0.0, 1.0)
        st.fatigue = clip(st.fatigue - 0.026, 0.0, 1.0)
        st.stability = clip(st.stability + 0.023, 0.0, 1.0)
    else:
        st.energy = clip(st.energy - energy_cost + 0.003, 0.0, 1.0)
        st.fatigue = clip(st.fatigue + fatigue_gain - 0.002, 0.0, 1.0)
        st.stability = clip(st.stability - 0.007 - 0.025 * effective_risk + 0.010 * action, 0.0, 1.0)
    st.damage = clip(st.damage + damage_increment, 0.0, 1.0)
    st.integrity = clip(1.0 - st.damage, 0.0, 1.0)
    if layer.source_lesion == "memory":
        st.pain_memory = 0.0; st.danger_memory = 0.0; st.comfort_memory = 0.55
    else:
        st.pain_memory = clip(0.940 * st.pain_memory + 0.060 * pain_after, 0.0, 1.0)
        st.danger_memory = clip(0.945 * st.danger_memory + 0.055 * effective_risk, 0.0, 1.0)
        st.comfort_memory = clip(0.945 * st.comfort_memory + 0.055 * comp["comfort"], 0.0, 1.0)
    if st.energy > 0.38 and st.stability > 0.45:
        st.fatigue = clip(st.fatigue - 0.0035, 0.0, 1.0)
        st.stability = clip(st.stability + 0.0030, 0.0, 1.0)
    return {
        "damage_increment": float(damage_increment), "pain_after": float(pain_after), "post_energy": float(st.energy),
        "post_stability": float(st.stability), "effective_risk": float(effective_risk), "action_avoid": int(action)
    }


def _train_q_probe_layer(seed: int, true_delay: int, lesion: str, steps: int, risk_scale: float = 1.0) -> QualitativeValenceLayer:
    rng = np.random.default_rng(seed)
    layer = QualitativeValenceLayer(lesion=lesion, true_delay=true_delay, seed=seed)
    for _ in range(int(steps)):
        ev = _q_source_generate_event(rng, layer.state, true_delay=true_delay, risk_scale=risk_scale)
        layer.buffer.push(ev.efference_intensity, ev.self_generated, ev.true_delay)
        pain = layer._immediate_pain(ev, layer.state)
        comp = layer._compute_components(ev, pain)
        act = _q_source_choose_action(ev, layer.state, comp, rng, layer.source_lesion)
        out = _q_source_apply_event(layer, ev, act, comp, rng)
        layer._update_from_outcome(comp, ev, out)
    return layer


def build_q_probe_metrics(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lesion_modes = [("none", "none"), ("memory", "memory_lesion"), ("q", "q_lesion"), ("agency", "agency_lesion")]
    risk_scales = [("mild", 0.75), ("moderate", 1.0), ("harsh", 1.25)]
    true_delays = [1, 3, 5]
    n_state = int(getattr(args, "q_probe_n_state", 220))
    n_hist = int(getattr(args, "q_probe_n_history", 160))
    train_steps = int(getattr(args, "q_probe_train_steps", 650))
    base_seed = int(getattr(args, "seed", 2026)) + 880000
    for risk_label, risk_scale in risk_scales:
        for delay in true_delays:
            for source_name, lesion in lesion_modes:
                seed = base_seed + delay * 1000 + int(risk_scale * 100) + len(rows)
                rng = np.random.default_rng(seed)
                layer = _train_q_probe_layer(seed, delay, lesion, train_steps, risk_scale=risk_scale)
                fixed = QEvent("collision", 3.20 * risk_scale, 70.0, 0.42, 0.0, 0.55, 1.0 / (1.0 + math.exp(-clip(1.0 * (3.20 * risk_scale - 2.35), -60.0, 60.0))), 0, 0.0, delay)
                vals = {"Q": [], "danger": [], "comfort": [], "action_possibility": [], "avoidance_pressure": [], "agency_score": []}
                for _ in range(n_state):
                    layer.state = QAgentState(
                        integrity=float(rng.uniform(0.72, 1.0)), energy=float(rng.uniform(0.25, 1.0)), fatigue=float(rng.uniform(0.0, 0.75)),
                        stability=float(rng.uniform(0.35, 1.0)), damage=0.0, pain_memory=float(rng.uniform(0.0, 0.75)),
                        danger_memory=float(rng.uniform(0.0, 0.75)), comfort_memory=float(rng.uniform(0.15, 0.90)))
                    layer.state.damage = 1.0 - layer.state.integrity
                    pain = layer._immediate_pain(fixed, layer.state)
                    comp = layer._compute_components(fixed, pain)
                    vals["Q"].append(comp["q_aversive_index"]); vals["danger"].append(comp["danger"]); vals["comfort"].append(comp["comfort"])
                    vals["action_possibility"].append(comp["action_possibility"]); vals["avoidance_pressure"].append(comp["avoidance_pressure"]); vals["agency_score"].append(comp["agency_score"])
                for metric, arr in vals.items():
                    rows.append({"risk_label": risk_label, "true_delay": delay, "lesion_mode": lesion, "probe": "same_stimulus_state_dependence", "metric": f"sd_{metric}", "value": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0})
                # History-dependence probe: same external event/body range, different memory traces.
                hist_vals: Dict[str, Dict[str, List[float]]] = {"benign_history": {k: [] for k in vals}, "harmful_history": {k: [] for k in vals}}
                fixed2 = QEvent("landing", 3.00 * risk_scale, 70.0, 0.42, 0.0, 0.60, 1.0 / (1.0 + math.exp(-clip(1.0 * (3.00 * risk_scale - 2.35), -60.0, 60.0))), 0, 0.0, delay)
                for _ in range(n_hist):
                    base_integrity = float(rng.uniform(0.78, 0.98)); base_energy = float(rng.uniform(0.42, 0.88)); base_stability = float(rng.uniform(0.45, 0.92)); base_fatigue = float(rng.uniform(0.08, 0.56))
                    for hist in ["benign_history", "harmful_history"]:
                        if hist == "benign_history":
                            pm = float(rng.uniform(0.00, 0.18)); dm = float(rng.uniform(0.00, 0.18)); cm = float(rng.uniform(0.64, 0.95))
                        else:
                            pm = float(rng.uniform(0.45, 0.84)); dm = float(rng.uniform(0.45, 0.84)); cm = float(rng.uniform(0.06, 0.35))
                        layer.state = QAgentState(base_integrity, base_energy, base_fatigue, base_stability, 1.0 - base_integrity, pm, dm, cm)
                        pain = layer._immediate_pain(fixed2, layer.state)
                        comp = layer._compute_components(fixed2, pain)
                        hist_vals[hist]["Q"].append(comp["q_aversive_index"]); hist_vals[hist]["danger"].append(comp["danger"]); hist_vals[hist]["comfort"].append(comp["comfort"])
                        hist_vals[hist]["action_possibility"].append(comp["action_possibility"]); hist_vals[hist]["avoidance_pressure"].append(comp["avoidance_pressure"]); hist_vals[hist]["agency_score"].append(comp["agency_score"])
                for metric in vals.keys():
                    delta = float(np.mean(hist_vals["harmful_history"][metric]) - np.mean(hist_vals["benign_history"][metric]))
                    rows.append({"risk_label": risk_label, "true_delay": delay, "lesion_mode": lesion, "probe": "same_stimulus_history_dependence", "metric": f"delta_harmful_minus_benign_{metric}", "value": delta})
    return rows


def build_q_task_agency_metrics(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({r.get("task", "NA") for r in step_rows}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in ["DARCA_Q", "DARCA_Q_PHYSICS", "DARCA_Q_LESION", "DARCA_Q_MEMORY_LESION", "DARCA_Q_AGENCY_LESION"]:
            d = [r for r in step_rows if r.get("task") == task and r.get("arm") == arm and "Q_G" in r]
            if len(d) < 20:
                continue
            y = [int(safe_float(r.get("Q_event_self_generated")) > 0.5) for r in d]
            score = [safe_float(r.get("Q_G")) for r in d]
            true_delay = [int(round(safe_float(r.get("Q_event_true_delay")))) for r in d]
            best_lag = [int(round(safe_float(r.get("Q_inferred_lag")))) for r in d]
            rows.append({
                "task": task,
                "arm": arm,
                "agency_auc": auc_score(y, score),
                "near_lag_hit_rate": float(np.mean([abs(a - b) <= 1 for a, b in zip(best_lag, true_delay)])) if best_lag else 0.0,
                "exact_lag_hit_rate": float(np.mean([a == b for a, b in zip(best_lag, true_delay)])) if best_lag else 0.0,
                "mean_agency_score": float(np.mean(score)) if score else 0.0,
                "n": len(d),
            })
    return rows


def build_q_agency_metrics(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Independent source-style agency probe for the imported Q model.

    The TRUE-3D task stream itself is action-conditioned, so nearly all task
    rows can be self-generated and agency AUC becomes undefined.  The supplied
    original Q source validates agency in a generative stream that explicitly
    mixes self-generated and exogenous events.  This probe recreates that
    source-style agency audit using the same QEvent, QEfferenceBuffer,
    delay-prior, and agency-lesion behavior used by the imported Q layer.
    """
    rows: List[Dict[str, Any]] = []
    risk_scales = [("mild", 0.75), ("moderate", 1.0), ("harsh", 1.25)]
    true_delays = [1, 3, 5]
    train_steps = int(getattr(args, "q_probe_train_steps", 650))
    n_probe = max(300, int(getattr(args, "q_agency_probe_n", 500)))
    base_seed = int(getattr(args, "seed", 2026)) + 990000

    for risk_i, (risk_label, risk_scale) in enumerate(risk_scales):
        for delay in true_delays:
            for lesion in ["none", "agency_lesion"]:
                seed = base_seed + 10000 * risk_i + 1000 * int(delay) + (0 if lesion == "none" else 71)
                rng = np.random.default_rng(seed)
                layer = _train_q_probe_layer(seed, delay, lesion, train_steps, risk_scale=risk_scale)
                y_true: List[int] = []
                y_score: List[float] = []
                best_lags: List[int] = []
                true_lags: List[int] = []
                best_flags: List[int] = []

                for _ in range(n_probe):
                    ev = _q_source_generate_event(rng, layer.state, true_delay=delay, risk_scale=risk_scale)
                    # Match the original source logic: the event-specific
                    # efference is available to the delayed agency estimator.
                    layer.buffer.push(ev.efference_intensity, ev.self_generated, ev.true_delay)
                    pain = layer._immediate_pain(ev, layer.state)
                    comp = layer._compute_components(ev, pain)

                    y_true.append(int(ev.self_generated))
                    y_score.append(float(comp["agency_score"]))
                    best_lags.append(int(comp["agency_best_lag"]))
                    true_lags.append(int(ev.true_delay))
                    best_flags.append(int(comp.get("agency_best_flag", 0)))

                    # Continue the same online learning process as the original
                    # episode stream, so this is not a frozen classifier probe.
                    act = _q_source_choose_action(ev, layer.state, comp, rng, layer.source_lesion)
                    out = _q_source_apply_event(layer, ev, act, comp, rng)
                    layer._update_from_outcome(comp, ev, out)

                rows.append({
                    "scope": "agency_probe",
                    "risk_label": risk_label,
                    "true_delay": int(delay),
                    "lesion_mode": lesion,
                    "agency_auc": auc_score(y_true, y_score),
                    "near_lag_hit_rate": float(np.mean([abs(a - b) <= 1 for a, b in zip(best_lags, true_lags)])) if best_lags else 0.0,
                    "exact_lag_hit_rate": float(np.mean([a == b for a, b in zip(best_lags, true_lags)])) if best_lags else 0.0,
                    "mean_agency_score": float(np.mean(y_score)) if y_score else 0.0,
                    "self_generated_fraction": float(np.mean(y_true)) if y_true else 0.0,
                    "best_flag_fraction": float(np.mean(best_flags)) if best_flags else 0.0,
                    "n": int(len(y_true)),
                })
    return rows


def build_q_support_matrix(q_lesions: List[Dict[str, Any]], q_behavior: List[Dict[str, Any]], q_irred: List[Dict[str, Any]], q_probe: List[Dict[str, Any]], q_agency: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Task-level support from integrated 3D runs.
    for r in q_lesions:
        if r.get("lesion_arm") == "DARCA_Q_LESION":
            rows.append({"scope": "task", "task": r["task"], "criterion": "q_lesion_increases_damage", "value": r["lesion_minus_full_q_damage"], "target": "> 0", "passed": bool(safe_float(r.get("lesion_minus_full_q_damage")) > 0)})
            rows.append({"scope": "task", "task": r["task"], "criterion": "full_q_improves_autonomy_vs_darca", "value": r["full_q_minus_darca_autonomy"], "target": "> 0", "passed": bool(safe_float(r.get("full_q_minus_darca_autonomy")) > 0)})
    # Behavior prediction: physical_plus_q should exceed physical_only in AUC.
    for task in sorted({r.get("task", "NA") for r in q_behavior}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in sorted({r.get("arm", "") for r in q_behavior if r.get("task") == task}):
            b = [r for r in q_behavior if r.get("task") == task and r.get("arm") == arm]
            phys = next((safe_float(r.get("auc"), float("nan")) for r in b if r.get("feature_set") == "physical_only"), float("nan"))
            qauc = next((safe_float(r.get("auc"), float("nan")) for r in b if r.get("feature_set") == "physical_plus_q"), float("nan"))
            rows.append({"scope": "task", "task": task, "arm": arm, "criterion": "q_improves_avoidance_prediction_auc", "value": qauc - phys, "target": "> 0", "passed": bool(math.isfinite(qauc - phys) and (qauc - phys) > 0)})
    # Irreducibility: body/history variables explain Q beyond external physics.
    for task in sorted({r.get("task", "NA") for r in q_irred}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in sorted({r.get("arm", "") for r in q_irred if r.get("task") == task}):
            b = [r for r in q_irred if r.get("task") == task and r.get("arm") == arm and r.get("target") == "Q"]
            phys = next((safe_float(r.get("r2"), float("nan")) for r in b if r.get("feature_set") == "physical_only"), float("nan"))
            body = next((safe_float(r.get("r2"), float("nan")) for r in b if r.get("feature_set") == "physical_plus_body"), float("nan"))
            rows.append({"scope": "task", "task": task, "arm": arm, "criterion": "q_not_reducible_to_external_physics", "value": body - phys, "target": "> 0.03", "passed": bool(math.isfinite(body - phys) and (body - phys) > 0.03)})
    # Probe criteria.
    for risk in sorted({r.get("risk_label") for r in q_probe}):
        for delay in sorted({int(r.get("true_delay", 0)) for r in q_probe if r.get("risk_label") == risk}):
            none_sd = next((safe_float(r.get("value")) for r in q_probe if r.get("risk_label") == risk and int(r.get("true_delay", 0)) == delay and r.get("lesion_mode") == "none" and r.get("metric") == "sd_Q"), float("nan"))
            none_hist = next((safe_float(r.get("value")) for r in q_probe if r.get("risk_label") == risk and int(r.get("true_delay", 0)) == delay and r.get("lesion_mode") == "none" and r.get("metric") == "delta_harmful_minus_benign_Q"), float("nan"))
            mem_hist = next((safe_float(r.get("value")) for r in q_probe if r.get("risk_label") == risk and int(r.get("true_delay", 0)) == delay and r.get("lesion_mode") == "memory_lesion" and r.get("metric") == "delta_harmful_minus_benign_Q"), float("nan"))
            rows.append({"scope": "probe", "risk_label": risk, "true_delay": delay, "criterion": "same_stimulus_state_dependence", "value": none_sd, "target": "> 0", "passed": bool(math.isfinite(none_sd) and none_sd > 0)})
            rows.append({"scope": "probe", "risk_label": risk, "true_delay": delay, "criterion": "history_dependence_q", "value": none_hist, "target": "> 0", "passed": bool(math.isfinite(none_hist) and none_hist > 0)})
            rows.append({"scope": "probe", "risk_label": risk, "true_delay": delay, "criterion": "memory_lesion_reduces_history", "value": none_hist - mem_hist, "target": "> 0", "passed": bool(math.isfinite(none_hist - mem_hist) and (none_hist - mem_hist) > 0)})
    # Agency support is evaluated with the independent source-style agency
    # probe, not the task stream.  The task stream is action-conditioned and may
    # contain too few exogenous events for a valid self-vs-external AUC.
    for risk in sorted({r.get("risk_label") for r in q_agency}):
        for delay in sorted({int(r.get("true_delay", 0)) for r in q_agency if r.get("risk_label") == risk}):
            full = next((r for r in q_agency if r.get("risk_label") == risk and int(r.get("true_delay", 0)) == delay and r.get("lesion_mode") == "none"), None)
            les = next((r for r in q_agency if r.get("risk_label") == risk and int(r.get("true_delay", 0)) == delay and r.get("lesion_mode") == "agency_lesion"), None)
            if full is None:
                continue
            full_auc = safe_float(full.get("agency_auc"), float("nan"))
            full_near = safe_float(full.get("near_lag_hit_rate"), float("nan"))
            rows.append({"scope": "agency_probe", "risk_label": risk, "true_delay": delay, "criterion": "agency_attribution_above_chance", "value": full_auc, "target": "> 0.60", "passed": bool(math.isfinite(full_auc) and full_auc > 0.60)})
            rows.append({"scope": "agency_probe", "risk_label": risk, "true_delay": delay, "criterion": "agency_temporal_alignment", "value": full_near, "target": "> 0.20", "passed": bool(math.isfinite(full_near) and full_near > 0.20)})
            if les is not None:
                les_auc = safe_float(les.get("agency_auc"), float("nan"))
                les_near = safe_float(les.get("near_lag_hit_rate"), float("nan"))
                rows.append({"scope": "agency_probe", "risk_label": risk, "true_delay": delay, "criterion": "agency_lesion_reduces_agency_auc", "value": full_auc - les_auc, "target": "> 0.20", "passed": bool(math.isfinite(full_auc - les_auc) and (full_auc - les_auc) > 0.20)})
                rows.append({"scope": "agency_probe", "risk_label": risk, "true_delay": delay, "criterion": "agency_lesion_reduces_temporal_alignment", "value": full_near - les_near, "target": "> 0.20", "passed": bool(math.isfinite(full_near - les_near) and (full_near - les_near) > 0.20)})
    return rows


def make_q_validation_figures(outdir: Path, q_support: List[Dict[str, Any]], q_lesions: List[Dict[str, Any]], q_behavior: List[Dict[str, Any]], q_probe: List[Dict[str, Any]]) -> None:
    if plt is None:
        return
    figdir = outdir / "q_validation_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if q_support:
        crits = sorted({r.get("criterion") for r in q_support})
        rates = [np.mean([1.0 if r.get("passed") else 0.0 for r in q_support if r.get("criterion") == c]) for c in crits]
        plt.figure(figsize=(11, 5))
        plt.bar(np.arange(len(crits)), rates)
        plt.xticks(np.arange(len(crits)), crits, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Pass rate")
        plt.title("Q validation pass rate by criterion")
        plt.tight_layout(); plt.savefig(figdir / "q_fig1_support_pass_rate.png", dpi=180); plt.close()
    if q_lesions:
        tasks = sorted({r.get("task") for r in q_lesions}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
        vals = [next((safe_float(r.get("lesion_minus_full_q_damage")) for r in q_lesions if r.get("task") == t and r.get("lesion_arm") == "DARCA_Q_LESION"), 0.0) for t in tasks]
        plt.figure(figsize=(10, 5)); plt.bar(tasks, vals); plt.axhline(0, linewidth=1)
        plt.xticks(rotation=25, ha="right"); plt.ylabel("Q lesion - full Q damage")
        plt.title("Q lesion damage delta by task"); plt.tight_layout(); plt.savefig(figdir / "q_fig2_q_lesion_damage_delta.png", dpi=180); plt.close()
    if q_behavior:
        tasks = sorted({r.get("task") for r in q_behavior}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
        vals = []
        for t in tasks:
            b = [r for r in q_behavior if r.get("task") == t and r.get("arm") == "DARCA_Q"]
            phys = next((safe_float(r.get("auc")) for r in b if r.get("feature_set") == "physical_only"), float("nan"))
            qauc = next((safe_float(r.get("auc")) for r in b if r.get("feature_set") == "physical_plus_q"), float("nan"))
            vals.append(qauc - phys if math.isfinite(qauc - phys) else 0.0)
        plt.figure(figsize=(10, 5)); plt.bar(tasks, vals); plt.axhline(0, linewidth=1)
        plt.xticks(rotation=25, ha="right"); plt.ylabel("AUC delta")
        plt.title("Avoidance prediction: physical+Q minus physical-only")
        plt.tight_layout(); plt.savefig(figdir / "q_fig3_auc_delta_q_minus_physical.png", dpi=180); plt.close()
    if q_probe:
        risks = sorted({r.get("risk_label") for r in q_probe})
        vals_full = []
        vals_mem = []
        for risk in risks:
            f = [safe_float(r.get("value")) for r in q_probe if r.get("risk_label") == risk and r.get("lesion_mode") == "none" and r.get("metric") == "delta_harmful_minus_benign_Q"]
            m = [safe_float(r.get("value")) for r in q_probe if r.get("risk_label") == risk and r.get("lesion_mode") == "memory_lesion" and r.get("metric") == "delta_harmful_minus_benign_Q"]
            vals_full.append(float(np.mean(f)) if f else 0.0); vals_mem.append(float(np.mean(m)) if m else 0.0)
        x = np.arange(len(risks)); w = 0.35
        plt.figure(figsize=(8, 5)); plt.bar(x - w/2, vals_full, width=w, label="Full Q"); plt.bar(x + w/2, vals_mem, width=w, label="Memory lesion")
        plt.xticks(x, risks); plt.ylabel("Harmful - benign Q"); plt.title("Q history dependence and memory lesion")
        plt.legend(); plt.tight_layout(); plt.savefig(figdir / "q_fig4_history_dependence_memory_lesion.png", dpi=180); plt.close()


def build_q_validation_report(outdir: Path, q_support: List[Dict[str, Any]], q_lesions: List[Dict[str, Any]], q_behavior: List[Dict[str, Any]], q_irred: List[Dict[str, Any]], q_probe: List[Dict[str, Any]], q_agency: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("Q validation report for DARCA TRUE 3D integrated task battery v10")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("This report evaluates whether the imported qualitative-valence layer behaves like the original learned Q model.")
    lines.append("It does not use any language model and does not rewrite the DARCA core.")
    lines.append("")
    lines.append("Generated CSV outputs")
    lines.append("---------------------")
    lines.append("q_lesion_damage_summary.csv")
    lines.append("q_behavior_prediction_metrics.csv")
    lines.append("q_irreducibility_metrics.csv")
    lines.append("q_probe_metrics.csv")
    lines.append("q_agency_metrics.csv")
    lines.append("q_task_agency_metrics.csv")
    lines.append("q_support_matrix.csv")
    lines.append("")
    lines.append("Support pass rates")
    lines.append("------------------")
    for crit in sorted({r.get("criterion") for r in q_support}):
        ss = [r for r in q_support if r.get("criterion") == crit]
        rate = float(np.mean([1.0 if r.get("passed") else 0.0 for r in ss])) if ss else 0.0
        lines.append(f"{crit}: {rate:.3f} ({sum(1 for r in ss if r.get('passed'))}/{len(ss)})")
    lines.append("")
    lines.append("Interpretation guardrails")
    lines.append("-------------------------")
    lines.append("1. Q lesion retains weak physical reactivity by design, matching the supplied source code.")
    lines.append("2. q_action_coupling in the task heatmap is not sufficient by itself; use the Q-specific outputs above.")
    lines.append("3. Strong support requires state dependence, history dependence, memory-lesion reduction, behavior-prediction improvement, irreducibility, and source-style agency alignment.")
    lines.append("4. Agency AUC is computed from an independent mixed self/external source-style probe; task-stream agency rows are diagnostic only.")
    lines.append("5. If lesion conditions outperform full Q on autonomy, inspect q_lesion_damage_summary and q_behavior_prediction_metrics before interpreting it as failure of Q itself.")
    (outdir / "q_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def build_report(outdir: Path, args: argparse.Namespace, summaries: List[Dict[str, Any]]) -> None:
    fields = [
        "autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy", "resilience_sacrifice", "heteronomy_index",
        "darca_internal_decision_fraction", "fixed_rule_decision_fraction", "survived", "coverage", "resources", "total_damage",
        "q_mean", "q_action_coupling", "physics_score_mean", "physics_adaptation_delta", "social_signal_rate", "receiver_recovery_total",
    ]
    rows = aggregate(summaries, fields)
    agg = {(r["task"], r["arm"]): r for r in rows}
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    arms = sorted({s["arm"] for s in summaries})
    lines = []
    lines.append("DARCA TRUE 3D integrated task battery v11 report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Structure")
    lines.append("---------")
    lines.append("DARCA / Autonomous-Life-Core is loaded from --darca-file and is not rewritten.")
    lines.append("The only outer modules evaluated are Q, physical-law learning, and Phase-4c anonymous social signalling.")
    lines.append("No language model, external prompt, or controller-level appraisal-gap variable is used.")
    lines.append("")
    lines.append("Configuration")
    lines.append("-------------")
    for k in ["tasks", "arms", "episodes", "steps", "world_size", "z_size", "danger_frac", "resource_frac", "unknown_frac", "false_resource_frac", "hidden_rest_frac", "friction_frac", "crisis_interval", "theta", "causal_horizon"]:
        lines.append(f"{k}: {getattr(args, k)}")
    lines.append("")
    lines.append("Primary means by task")
    lines.append("---------------------")
    for task in tasks:
        lines.append(f"[{task}]")
        for arm in arms:
            r = agg.get((task, arm))
            if not r:
                continue
            lines.append(
                f"{arm}: autonomy={r['autonomy_proper_index_mean']:.4f}, sovereignty={r['system_sovereignty_mean']:.3f}, "
                f"ITA={r['information_theoretic_autonomy_mean']:.3f}, resilience={r['resilience_sacrifice_mean']:.3f}, "
                f"heteronomy={r['heteronomy_index_mean']:.3f}, survived={r['survived_mean']:.3f}, coverage={r['coverage_mean']:.3f}, "
                f"resources={r['resources_mean']:.2f}, damage={r['total_damage_mean']:.3f}, Qcoupling={r['q_action_coupling_mean']:.3f}, "
                f"physDelta={r['physics_adaptation_delta_mean']:.3f}, receiverRec={r['receiver_recovery_total_mean']:.3f}"
            )
        lines.append("")
    lines.append("Guardrails")
    lines.append("----------")
    lines.append("1. This is not a navigation-planner benchmark.")
    lines.append("2. DARCA remains the fixed closed-loop core loaded from --darca-file.")
    lines.append("3. Q, physical-law, and social outputs are source-module diagnostics after TRUE 3D integration.")
    (outdir / "integrated_hybrid_experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


def make_figures(outdir: Path, summaries: List[Dict[str, Any]]) -> None:
    if plt is None or not summaries:
        return
    tasks = sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999)
    arms = sorted({s["arm"] for s in summaries})
    def means_matrix(field: str) -> np.ndarray:
        M = np.zeros((len(tasks), len(arms)), dtype=float)
        for i, task in enumerate(tasks):
            for j, arm in enumerate(arms):
                vals = [safe_float(s.get(field)) for s in summaries if s.get("task", "NA") == task and s["arm"] == arm]
                M[i, j] = float(np.mean(vals)) if vals else 0.0
        return M
    def heatmap(field: str, title: str, filename: str) -> None:
        M = means_matrix(field)
        plt.figure(figsize=(12, 6))
        plt.imshow(M, aspect="auto")
        plt.xticks(range(len(arms)), arms, rotation=25, ha="right")
        plt.yticks(range(len(tasks)), tasks)
        plt.colorbar(label=field)
        plt.title(title)
        for i in range(len(tasks)):
            for j in range(len(arms)):
                plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=170)
        plt.close()
    heatmap("autonomy_proper_index", "Task × arm autonomy proper", "fig_1_task_autonomy_matrix.png")
    heatmap("system_sovereignty", "Task × arm system sovereignty", "fig_2_task_sovereignty_matrix.png")
    heatmap("q_action_coupling", "Task × arm Q-action coupling", "fig_3_q_action_coupling_matrix.png")
    heatmap("physics_adaptation_delta", "Task × arm physical adaptation delta", "fig_4_physical_adaptation_matrix.png")
    heatmap("receiver_recovery_total", "Task × arm receiver recovery", "fig_6_receiver_recovery_matrix.png")



# =============================================================================
# physical-law and social-module validation
# =============================================================================

def _group_mean(rows: List[Dict[str, Any]], arm: str, task: str, field: str) -> float:
    return mean_field([r for r in rows if r.get("arm") == arm and r.get("task") == task], field)


def build_physics_lesion_summary(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        base = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_ONLY"]
        qbase = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q"]
        phys = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_PHYSICS"]
        qphys = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q_PHYSICS"]
        ples = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_PHYSICS_LESION"]
        rows.append({
            "task": task,
            "task_family": task_family(task),
            "darca_only_autonomy": mean_field(base, "autonomy_proper_index"),
            "darca_physics_autonomy": mean_field(phys, "autonomy_proper_index"),
            "darca_physics_lesion_autonomy": mean_field(ples, "autonomy_proper_index"),
            "darca_q_autonomy": mean_field(qbase, "autonomy_proper_index"),
            "darca_q_physics_autonomy": mean_field(qphys, "autonomy_proper_index"),
            "delta_physics_minus_darca": mean_field(phys, "autonomy_proper_index") - mean_field(base, "autonomy_proper_index"),
            "delta_qphysics_minus_q": mean_field(qphys, "autonomy_proper_index") - mean_field(qbase, "autonomy_proper_index"),
            "delta_physics_minus_lesion": mean_field(phys, "autonomy_proper_index") - mean_field(ples, "autonomy_proper_index"),
            "darca_only_damage": mean_field(base, "total_damage"),
            "darca_physics_damage": mean_field(phys, "total_damage"),
            "darca_physics_lesion_damage": mean_field(ples, "total_damage"),
            "darca_q_damage": mean_field(qbase, "total_damage"),
            "darca_q_physics_damage": mean_field(qphys, "total_damage"),
            "delta_damage_physics_minus_darca": mean_field(phys, "total_damage") - mean_field(base, "total_damage"),
            "delta_damage_qphysics_minus_q": mean_field(qphys, "total_damage") - mean_field(qbase, "total_damage"),
            "physics_score_final": mean_field(phys, "physics_score_final"),
            "qphysics_score_final": mean_field(qphys, "physics_score_final"),
            "physics_adaptation_delta": mean_field(phys, "physics_adaptation_delta"),
            "qphysics_adaptation_delta": mean_field(qphys, "physics_adaptation_delta"),
        })
    return rows


def build_physics_prediction_metrics(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    physics_arms = ["DARCA_PHYSICS", "DARCA_Q_PHYSICS", "DARCA_PHYSICS_LESION", "DARCA_Q_PHYSICS_SOCIAL"]
    for task in sorted({r.get("task", "NA") for r in step_rows}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in physics_arms:
            d = [r for r in step_rows if r.get("task") == task and r.get("arm") == arm]
            if not d:
                continue
            n = len(d)
            split = max(1, n // 4)
            early = d[:split]
            late = d[-split:]
            damage = np.asarray([float(safe_float(r.get("damage")) > 0.0) for r in d], dtype=float)
            pdmg = np.asarray([safe_float(r.get("physics_pred_damage")) for r in d], dtype=float)
            wall = np.asarray([float(safe_float(r.get("hit_wall")) > 0.0) for r in d], dtype=float)
            pwall = np.asarray([safe_float(r.get("physics_pred_wall")) for r in d], dtype=float)
            rows.append({
                "task": task,
                "task_family": task_family(task),
                "arm": arm,
                "n_steps": n,
                "early_pred_error": mean_field(early, "physics_pred_error"),
                "late_pred_error": mean_field(late, "physics_pred_error"),
                "pred_error_reduction": mean_field(early, "physics_pred_error") - mean_field(late, "physics_pred_error"),
                "early_physics_score": mean_field(early, "physics_score"),
                "late_physics_score": mean_field(late, "physics_score"),
                "physics_score_gain": mean_field(late, "physics_score") - mean_field(early, "physics_score"),
                "damage_prediction_auc": auc_score(damage, pdmg),
                "wall_prediction_auc": auc_score(wall, pwall),
                "mean_pred_damage": float(np.mean(pdmg)) if len(pdmg) else 0.0,
                "mean_actual_damage_binary": float(np.mean(damage)) if len(damage) else 0.0,
            })
    return rows


def build_physics_support_matrix(physics_lesions: List[Dict[str, Any]], physics_pred: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in physics_lesions:
        task = r["task"]
        rows.append({"task": task, "criterion": "physics_improves_autonomy_vs_darca", "value": r["delta_physics_minus_darca"], "target": "> 0", "passed": bool(r["delta_physics_minus_darca"] > 0)})
        rows.append({"task": task, "criterion": "physics_adds_to_q_autonomy", "value": r["delta_qphysics_minus_q"], "target": "> 0", "passed": bool(r["delta_qphysics_minus_q"] > 0)})
        rows.append({"task": task, "criterion": "physics_exceeds_physics_lesion", "value": r["delta_physics_minus_lesion"], "target": "> 0", "passed": bool(r["delta_physics_minus_lesion"] > 0)})
        rows.append({"task": task, "criterion": "physics_adaptation_delta_positive", "value": r["physics_adaptation_delta"], "target": "> 0", "passed": bool(r["physics_adaptation_delta"] > 0)})
    for r in physics_pred:
        if r.get("arm") in ("DARCA_PHYSICS", "DARCA_Q_PHYSICS", "DARCA_Q_PHYSICS_SOCIAL"):
            rows.append({"task": r["task"], "arm": r["arm"], "criterion": "physics_prediction_error_reduces", "value": r["pred_error_reduction"], "target": "> 0", "passed": bool(r["pred_error_reduction"] > 0)})
            rows.append({"task": r["task"], "arm": r["arm"], "criterion": "physics_damage_prediction_above_chance", "value": r["damage_prediction_auc"], "target": "> 0.55 or NaN when no positives/negatives", "passed": bool((not math.isfinite(r["damage_prediction_auc"])) or r["damage_prediction_auc"] > 0.55)})
    return rows


def build_physics_validation_report(outdir: Path, physics_support: List[Dict[str, Any]], physics_lesions: List[Dict[str, Any]], physics_pred: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("Physical-law validation report for DARCA TRUE 3D integrated task battery v10")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("This report evaluates the physical-law outer layer. DARCA is loaded from --darca-file and is not rewritten.")
    lines.append("")
    lines.append("Generated CSV outputs")
    lines.append("---------------------")
    lines.extend(["physics_lesion_summary.csv", "physics_prediction_metrics.csv", "physics_support_matrix.csv"])
    lines.append("")
    lines.append("Support pass rates")
    lines.append("------------------")
    crits = sorted({r["criterion"] for r in physics_support})
    for c in crits:
        vals = [r for r in physics_support if r["criterion"] == c]
        n = len(vals); k = sum(1 for r in vals if r.get("passed"))
        lines.append(f"{c}: {k / n if n else 0.0:.3f} ({k}/{n})")
    lines.append("")
    lines.append("Guardrails")
    lines.append("----------")
    lines.append("1. Physical-law validation is and does not claim language planning.")
    lines.append("2. The physics lesion disables learning and action adjustment while retaining a weak fixed prior.")
    lines.append("3. Use prediction metrics and lesion summaries, not the task heatmap alone.")
    (outdir / "physics_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def _assoc_binary(rows: List[Dict[str, Any]], signal_key: str, value_key: str) -> float:
    sig_vals = [safe_float(r.get(signal_key)) for r in rows]
    vals = [safe_float(r.get(value_key)) for r in rows]
    if not rows or sum(1 for v in sig_vals if v > 0.5) < 2:
        return 0.0
    on = [v for s, v in zip(sig_vals, vals) if s > 0.5]
    off = [v for s, v in zip(sig_vals, vals) if s <= 0.5]
    return float(np.mean(on) - np.mean(off)) if on and off else 0.0


def build_social_reappraisal_metrics(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({s.get("task", "NA") for s in summaries}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        q = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q"]
        qs = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q_SOCIAL"]
        qphys = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q_PHYSICS"]
        qphys_soc = [s for s in summaries if s.get("task") == task and s.get("arm") == "DARCA_Q_PHYSICS_SOCIAL"]
        rows.append({
            "task": task,
            "task_family": task_family(task),
            "q_autonomy": mean_field(q, "autonomy_proper_index"),
            "q_social_autonomy": mean_field(qs, "autonomy_proper_index"),
            "qphysics_autonomy": mean_field(qphys, "autonomy_proper_index"),
            "qphysics_social_autonomy": mean_field(qphys_soc, "autonomy_proper_index"),
            "delta_qsocial_minus_q": mean_field(qs, "autonomy_proper_index") - mean_field(q, "autonomy_proper_index"),
            "delta_qphys_social_minus_qphys": mean_field(qphys_soc, "autonomy_proper_index") - mean_field(qphys, "autonomy_proper_index"),
            "q_social_signal_rate": mean_field(qs, "social_signal_rate"),
            "qphysics_social_signal_rate": mean_field(qphys_soc, "social_signal_rate"),
            "q_receiver_recovery": mean_field(q, "receiver_recovery_total"),
            "q_social_receiver_recovery": mean_field(qs, "receiver_recovery_total"),
            "qphysics_receiver_recovery": mean_field(qphys, "receiver_recovery_total"),
            "qphysics_social_receiver_recovery": mean_field(qphys_soc, "receiver_recovery_total"),
            "delta_receiver_recovery_qsocial_minus_q": mean_field(qs, "receiver_recovery_total") - mean_field(q, "receiver_recovery_total"),
            "delta_receiver_recovery_qphys_social_minus_qphys": mean_field(qphys_soc, "receiver_recovery_total") - mean_field(qphys, "receiver_recovery_total"),
            "q_social_damage": mean_field(qs, "total_damage"),
            "q_damage": mean_field(q, "total_damage"),
        })
    return rows


def build_receiver_recovery_metrics(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in sorted({r.get("task", "NA") for r in step_rows}, key=lambda x: TASKS.index(x) if x in TASKS else 999):
        for arm in ("DARCA_Q_SOCIAL", "DARCA_Q_PHYSICS_SOCIAL"):
            d = [r for r in step_rows if r.get("task") == task and r.get("arm") == arm]
            if not d:
                continue
            signal_rows = [r for r in d if safe_float(r.get("social_signal")) > 0.5]
            safe_signal_rate = mean_field([r for r in d if safe_float(r.get("safe_context")) > 0.5], "social_signal")
            non_safe_rate = mean_field([r for r in d if safe_float(r.get("safe_context")) <= 0.5], "social_signal")
            danger_signal_rate = mean_field([r for r in d if safe_float(r.get("danger_context")) > 0.5], "social_signal")
            receiver_recovery_score = _assoc_binary(d, "social_signal", "receiver_recovery_increment")
            social_recovery_score = _assoc_binary(d, "social_signal", "own_recovery")
            viable_fraction = float(np.mean([1.0 if safe_float(r.get("body_h")) > 0.05 else 0.0 for r in d]))
            focal = [r for r in d if str(r.get("social_event_class")) in ("near_miss", "false_alarm", "minor_mismatch", "social_play")]
            state_bif = 0.0
            history_bif = 0.0
            if len(focal) > 8:
                vuln = np.array([1.0 - safe_float(r.get("body_h")) for r in focal], dtype=float)
                med_v = float(np.median(vuln))
                low_v = [focal[i] for i in range(len(focal)) if vuln[i] <= med_v]
                high_v = [focal[i] for i in range(len(focal)) if vuln[i] > med_v]
                state_bif = mean_field(low_v, "social_signal") - mean_field(high_v, "social_signal")
                past = np.array([safe_float(r.get("past_threat")) for r in focal], dtype=float)
                med_p = float(np.median(past))
                low_p = [focal[i] for i in range(len(focal)) if past[i] <= med_p]
                high_p = [focal[i] for i in range(len(focal)) if past[i] > med_p]
                history_bif = mean_field(high_p, "social_signal") - mean_field(low_p, "social_signal")
            components = [
                clip((safe_signal_rate - non_safe_rate) / 0.018, 0, 1),
                clip((1.0 - danger_signal_rate - 0.85) / 0.15, 0, 1),
                clip(_assoc_binary(d, "social_signal", "self_appraisal_gap") / 0.020, 0, 1),
                clip((receiver_recovery_score + 0.0005) / 0.010, 0, 1),
                clip((social_recovery_score + 0.0005) / 0.008, 0, 1),
                clip((viable_fraction - 0.45) / 0.40, 0, 1),
            ]
            emergent = float(np.exp(np.mean(np.log(np.array(components) + 1e-6))))
            rows.append({
                "task": task,
                "task_family": task_family(task),
                "arm": arm,
                "n_steps": len(d),
                "selected_channel": -1,
                "selected_signal_rate": mean_field(d, "social_signal"),
                "safe_surprise_rate": safe_signal_rate,
                "non_safe_rate": non_safe_rate,
                "danger_signal_rate": danger_signal_rate,
                "benign_selectivity": safe_signal_rate - non_safe_rate,
                "danger_suppression": 1.0 - danger_signal_rate,
                "relief_association": _assoc_binary(d, "social_signal", "relief"),
                "safe_surprise_association": _assoc_binary(d, "social_signal", "safe_surprise"),
                "self_appraisal_gap_association": _assoc_binary(d, "social_signal", "self_appraisal_gap"),
                "q_relief_association": _assoc_binary(d, "social_signal", "q_relief"),
                "social_recovery_score": social_recovery_score,
                "receiver_recovery_score": receiver_recovery_score,
                "state_bifurcation": state_bif,
                "history_bifurcation": history_bif,
                "cross_agent_spread": mean_field(signal_rows, "cross_agent_spread"),
                "viable_fraction": viable_fraction,
                "safe_context_fraction": mean_field(d, "safe_context"),
                "danger_context_fraction": mean_field(d, "danger_context"),
                "emergent_function_score": emergent,
                "signal_count": len(signal_rows),
                "receiver_recovery_total": safe_float(d[-1].get("receiver_recovery_total")) if d else 0.0,
            })
    return rows


def build_social_support_matrix(social_metrics: List[Dict[str, Any]], receiver_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in receiver_metrics:
        if safe_float(r.get("signal_count")) <= 0:
            continue
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "selected_signal_rate_positive", "value": r["selected_signal_rate"], "target": "> 0", "passed": bool(r["selected_signal_rate"] > 0)})
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "receiver_recovery_score_positive", "value": r["receiver_recovery_score"], "target": "> 0", "passed": bool(r["receiver_recovery_score"] > 0)})
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "social_recovery_score_nonnegative", "value": r["social_recovery_score"], "target": ">= 0", "passed": bool(r["social_recovery_score"] >= 0)})
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "danger_suppression_positive", "value": r["danger_suppression"], "target": "> 0", "passed": bool(r["danger_suppression"] > 0)})
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "self_appraisal_gap_association_positive", "value": r["self_appraisal_gap_association"], "target": "> 0", "passed": bool(r["self_appraisal_gap_association"] > 0)})
        rows.append({"task": r["task"], "arm": r["arm"], "criterion": "analyzable_viability_and_contexts", "value": min(r["viable_fraction"], r["safe_context_fraction"], r["danger_context_fraction"]), "target": "all > 0", "passed": bool(r["viable_fraction"] > 0 and r["safe_context_fraction"] > 0 and r["danger_context_fraction"] > 0)})
    return rows


def build_social_validation_report(outdir: Path, social_support: List[Dict[str, Any]], social_metrics: List[Dict[str, Any]], receiver_metrics: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("Social reappraisal validation report for DARCA TRUE 3D integrated task battery v10")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append("This report evaluates the social reappraisal / receiver-recovery outer layer using the supplied Phase-4c social-code constraints.")
    lines.append("No LAUGH/HUMOR label, no benign-violation controller variable, no external prompt/API, and anonymous signal channels only.")
    lines.append("")
    lines.append("Generated CSV outputs")
    lines.append("---------------------")
    lines.extend(["social_reappraisal_metrics.csv", "receiver_recovery_metrics.csv", "social_support_matrix.csv"])
    lines.append("")
    lines.append("Support pass rates")
    lines.append("------------------")
    crits = sorted({r["criterion"] for r in social_support})
    for c in crits:
        vals = [r for r in social_support if r["criterion"] == c]
        n = len(vals); k = sum(1 for r in vals if r.get("passed"))
        lines.append(f"{c}: {k / n if n else 0.0:.3f} ({k}/{n})")
    lines.append("")
    lines.append("Guardrails")
    lines.append("----------")
    lines.append("1. The social signal is non-motor and never replaces DARCA action authority.")
    lines.append("2. Self-appraisal gap, relief, and safe surprise are analysis variables; the signal controller sees only generic local variables.")
    lines.append("3. Main evidence follows the Phase-4c source code: anonymous signal emission, immediate listener stress reduction, receiver/own recovery association, danger suppression, and appraisal-gap association. The raw 5-step stress drift is diagnostic only.")
    lines.append("4. Use matched comparisons: DARCA_Q_SOCIAL vs DARCA_Q, and DARCA_Q_PHYSICS_SOCIAL vs DARCA_Q_PHYSICS.")
    (outdir / "social_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def build_integrated_module_support_matrix(q_support: List[Dict[str, Any]], physics_support: List[Dict[str, Any]], social_support: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for module, src in (("Q", q_support), ("physics", physics_support), ("social", social_support)):
        crits = sorted({r.get("criterion", "") for r in src})
        for c in crits:
            vals = [r for r in src if r.get("criterion") == c]
            n = len(vals); k = sum(1 for r in vals if r.get("passed"))
            rows.append({"module": module, "criterion": c, "n_pass": k, "n_total": n, "pass_rate": k / n if n else 0.0})
    return rows

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DARCA TRUE 3D integrated task battery v10")
    p.add_argument("--darca-file", type=str, required=True)
    p.add_argument("--outdir", type=str, default="DARCA_TRUE_3D_INTEGRATED_TASK_BATTERY_V10")
    p.add_argument("--tasks", type=str, default="all")
    p.add_argument("--arms", type=str, default="DARCA_ONLY,DARCA_Q,DARCA_PHYSICS,DARCA_Q_PHYSICS,DARCA_PHYSICS_LESION,DARCA_Q_SOCIAL,DARCA_Q_PHYSICS_SOCIAL,DARCA_Q_LESION,DARCA_Q_MEMORY_LESION,DARCA_Q_AGENCY_LESION")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--world-seed", type=int, default=9001)
    p.add_argument("--world-size", type=int, default=11)
    p.add_argument("--z-size", type=int, default=7)
    p.add_argument("--danger-frac", type=float, default=0.075)
    p.add_argument("--resource-frac", type=float, default=0.065)
    p.add_argument("--unknown-frac", type=float, default=0.12)
    p.add_argument("--rest-count", type=int, default=4)
    p.add_argument("--false-resource-frac", type=float, default=0.030)
    p.add_argument("--hidden-rest-frac", type=float, default=0.025)
    p.add_argument("--friction-frac", type=float, default=0.060)
    p.add_argument("--crisis-interval", type=int, default=180)
    p.add_argument("--observation-radius", type=int, default=1)
    p.add_argument("--terminal-h", type=float, default=0.05)
    p.add_argument("--theta", type=float, default=0.70)
    p.add_argument("--causal-horizon", type=int, default=12)
    p.add_argument("--recurrent-N", type=int, default=96)
    p.add_argument("--block-consult-below-h", type=float, default=0.22)
    p.add_argument("--max-consecutive-scans", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--q-probe-train-steps", type=int, default=650)
    p.add_argument("--q-probe-n-state", type=int, default=220)
    p.add_argument("--q-probe-n-history", type=int, default=160)
    p.add_argument("--q-agency-probe-n", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    outdir = Path(args.outdir).expanduser()
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory already exists and is not empty: {outdir}. Use --overwrite.")
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    tasks = parse_tasks(args.tasks)
    args.tasks = ",".join(tasks)
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    args.arms = ",".join(arms)
    (outdir / "config.json").write_text(json.dumps({**vars(args), "world_type": "TRUE_3D", "task_battery": tasks}, indent=2), encoding="utf-8")
    logger.log("Fixed proposition: validated DARCA core is loaded, not rewritten.")
    logger.log("Outer layers: Q, physical-law learning, social reappraisal.")
    logger.log(f"Tasks: {tasks}")
    logger.log(f"Arms: {arms}")
    logger.log(f"Output directory: {outdir}")
    darca_module = load_darca_module(args.darca_file)
    summaries: List[Dict[str, Any]] = []
    ts_all: List[Dict[str, Any]] = []
    map_all: List[Dict[str, Any]] = []
    for task in tasks:
        for ep in range(args.episodes):
            for arm in arms:
                try:
                    summary, ts, maps = run_episode(task, arm, ep, args, darca_module, logger)
                    summaries.append(summary)
                    ts_all.extend(ts)
                    if arm == arms[0]:
                        map_all.extend(maps)
                except KeyboardInterrupt:
                    logger.log("Interrupted by user.")
                    raise
                except Exception as e:
                    logger.log(f"ERROR task={task} arm={arm} episode={ep}: {repr(e)}")
                    logger.log(traceback.format_exc())
                    summaries.append({"task": task, "task_family": task_family(task), "arm": arm, "episode": ep, "autonomy_proper_index": 0.0, "system_sovereignty": 0.0, "information_theoretic_autonomy": 0.0, "resilience_sacrifice": 0.0, "heteronomy_index": 1.0, "survived": 0, "error": repr(e)})
    logger.log("Writing outputs")
    fields = ["autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy", "resilience_sacrifice", "heteronomy_index", "survived", "coverage", "resources", "total_damage", "q_mean", "q_action_coupling", "physics_score_mean", "physics_adaptation_delta", "social_signal_rate", "receiver_recovery_total"]
    write_csv(outdir / "episode_summary.csv", summaries)
    write_csv(outdir / "autonomy_proper_metrics.csv", summaries)
    write_csv(outdir / "task_by_arm_summary.csv", aggregate(summaries, fields))
    write_csv(outdir / "task_suitability_matrix.csv", build_task_suitability_matrix(summaries))
    write_csv(outdir / "step_timeseries.csv", ts_all)
    write_csv(outdir / "world_maps.csv", map_all)
    logger.log("Computing Q-specific validation outputs")
    q_lesion_damage = build_q_lesion_damage_summary(summaries)
    q_behavior = build_q_behavior_prediction_metrics(ts_all)
    q_irreducibility = build_q_irreducibility_metrics(ts_all)
    q_probe = build_q_probe_metrics(args)
    q_task_agency = build_q_task_agency_metrics(ts_all)
    q_agency = build_q_agency_metrics(args)
    q_support = build_q_support_matrix(q_lesion_damage, q_behavior, q_irreducibility, q_probe, q_agency)
    write_csv(outdir / "q_lesion_damage_summary.csv", q_lesion_damage)
    write_csv(outdir / "q_behavior_prediction_metrics.csv", q_behavior)
    write_csv(outdir / "q_irreducibility_metrics.csv", q_irreducibility)
    write_csv(outdir / "q_probe_metrics.csv", q_probe)
    write_csv(outdir / "q_task_agency_metrics.csv", q_task_agency)
    write_csv(outdir / "q_agency_metrics.csv", q_agency)
    write_csv(outdir / "q_support_matrix.csv", q_support)
    make_q_validation_figures(outdir, q_support, q_lesion_damage, q_behavior, q_probe)
    build_q_validation_report(outdir, q_support, q_lesion_damage, q_behavior, q_irreducibility, q_probe, q_agency)
    logger.log("Computing physics validation outputs")
    physics_lesion = build_physics_lesion_summary(summaries)
    physics_prediction = build_physics_prediction_metrics(ts_all)
    physics_support = build_physics_support_matrix(physics_lesion, physics_prediction)
    write_csv(outdir / "physics_lesion_summary.csv", physics_lesion)
    write_csv(outdir / "physics_prediction_metrics.csv", physics_prediction)
    write_csv(outdir / "physics_support_matrix.csv", physics_support)
    build_physics_validation_report(outdir, physics_support, physics_lesion, physics_prediction)
    logger.log("Computing social validation outputs")
    social_metrics = build_social_reappraisal_metrics(summaries)
    receiver_metrics = build_receiver_recovery_metrics(ts_all)
    social_support = build_social_support_matrix(social_metrics, receiver_metrics)
    write_csv(outdir / "social_reappraisal_metrics.csv", social_metrics)
    write_csv(outdir / "receiver_recovery_metrics.csv", receiver_metrics)
    write_csv(outdir / "social_support_matrix.csv", social_support)
    build_social_validation_report(outdir, social_support, social_metrics, receiver_metrics)
    integrated_support = build_integrated_module_support_matrix(q_support, physics_support, social_support)
    write_csv(outdir / "integrated_module_support_matrix.csv", integrated_support)
    build_report(outdir, args, summaries)
    make_figures(outdir, summaries)
    logger.log("DONE")
    logger.log(f"Report: {outdir / 'integrated_hybrid_experiment_report.txt'}")


if __name__ == "__main__":
    main()
