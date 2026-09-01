#!/usr/bin/env python3.8
"""
rospug_env_adr.py — Sensitivity-Weighted Adaptive Domain Randomization (Step 6).

Novel contribution: per-parameter return-variance at range boundaries directs
differential curriculum expansion.  Most brittle physics dimension expands fastest.

Standard ADR (Akkaya et al. 2019) expands all ranges at the same rate based on
aggregate success rate.  SW-ADR instead measures, for each parameter independently:

    sensitivity_d = Var(returns | d sampled at boundary)
                  - Var(returns | d sampled at interior)

A high boundary variance means the policy is unstable at the edge of that
dimension — it is brittle there.  Range expansion is allocated proportionally to
normalised sensitivity weights, so the curriculum automatically targets weakness.

Class hierarchy (zero changes to parent classes):
    RosPugEnv
      └─ RosPugEnvDR   (fixed ranges — Policy B)
           └─ RosPugEnvADR  (adaptive ranges — Policy C)

Key design: reset() calls RosPugEnv.reset() directly (grandparent bypass) to
avoid RosPugEnvDR.reset() which would re-sample from the fixed constructor ranges.
All Gazebo physics helpers (_set_body_mass, _set_ode_physics, _publish_joints
latency injection) are inherited and used unchanged.

Usage::

    env = RosPugEnvADR()
    obs, info = env.reset()
    print(info['adr_ranges'])   # current [lo, hi] per dimension
    # After 50 episodes the ranges start evolving

    # Held-out evaluation (fixed ranges outside training bounds — for Step 7):
    env = RosPugEnvADR(mass_range=(1.20, 1.20), latency_range=(0.015, 0.015))
"""

from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

import rospy

from rl_env.rospug_env import RosPugEnv
from rl_env.rospug_env_dr import RosPugEnvDR


class RosPugEnvADR(RosPugEnvDR):
    """
    RosPugEnvDR with Sensitivity-Weighted Adaptive Domain Randomization.

    Parameters
    ----------
    mass_range, latency_range, cfm_range, erp_range :
        Starting ranges — identical to Policy B for a fair comparison baseline.
    update_interval : episodes between sensitivity computation and range update.
    buffer_size : rolling window length (≥ update_interval recommended).
    boundary_frac : fraction of range width defining the boundary zone (top + bottom).
    min_boundary_eps : minimum boundary-zone episodes required for a valid sensitivity
        estimate; dimension treated as non-brittle if below this count.
    """

    # Hard caps — ranges never expand beyond these bounds
    _MAX_RANGES: Dict[str, List[float]] = {
        'mass':    [0.70, 1.30],
        'latency': [0.00, 0.020],
        'cfm':     [0.00, 0.002],
        'erp':     [0.10, 0.90],
    }

    # Maximum half-width expansion per update if a dimension has full sensitivity weight
    _EXPAND_BUDGET: Dict[str, float] = {
        'mass':    0.02,
        'latency': 0.002,
        'cfm':     0.0002,
        'erp':     0.05,
    }

    _DIMS = ('mass', 'latency', 'cfm', 'erp')

    def __init__(
        self,
        mass_range: Tuple[float, float]    = (0.85, 1.15),
        latency_range: Tuple[float, float] = (0.0, 0.010),
        cfm_range: Tuple[float, float]     = (0.0, 0.0005),
        erp_range: Tuple[float, float]     = (0.3, 0.8),
        update_interval: int  = 50,
        buffer_size: int      = 50,
        boundary_frac: float  = 0.20,
        min_boundary_eps: int = 5,
        node_name: str = 'rospug_env_adr',
    ) -> None:
        super().__init__(
            mass_range=mass_range,
            latency_range=latency_range,
            cfm_range=cfm_range,
            erp_range=erp_range,
            node_name=node_name,
        )

        self._update_interval  = update_interval
        self._boundary_frac    = boundary_frac
        self._min_boundary_eps = min_boundary_eps

        # Live ranges that the ADR curriculum expands over training
        self._curr_ranges: Dict[str, List[float]] = {
            'mass':    list(mass_range),
            'latency': list(latency_range),
            'cfm':     list(cfm_range),
            'erp':     list(erp_range),
        }

        # Rolling buffer: {'params': {dim: value, ...}, 'return': float}
        self._buffer: deque = deque(maxlen=buffer_size)

        self._episode_count:   int   = 0
        self._update_count:    int   = 0
        self._episode_return:  float = 0.0
        self._ep_raw:          Dict[str, float] = {}

        # Most recent sensitivity and weight values — 0/uniform until first update
        self._last_sensitivities: Dict[str, float] = {d: 0.0  for d in self._DIMS}
        self._last_weights:       Dict[str, float] = {d: 0.25 for d in self._DIMS}

        rospy.loginfo(
            'RosPugEnvADR ready: update_interval=%d  buffer=%d  boundary_frac=%.2f',
            update_interval, buffer_size, boundary_frac,
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Sample from current (evolving) ranges, apply physics, call grandparent."""

        mass_frac = float(np.random.uniform(*self._curr_ranges['mass']))
        latency   = float(np.random.uniform(*self._curr_ranges['latency']))
        cfm       = float(np.random.uniform(*self._curr_ranges['cfm']))
        erp       = float(np.random.uniform(*self._curr_ranges['erp']))

        # Physics helpers inherited from RosPugEnvDR
        self._set_body_mass(mass_frac)
        self._set_ode_physics(cfm, erp)

        # Fields read by inherited _publish_joints / step()
        self._episode_latency = latency
        self._dr_params = {
            'mass_frac':  round(mass_frac, 4),
            'latency_ms': round(latency * 1000.0, 2),
            'cfm':        round(cfm, 6),
            'erp':        round(erp, 3),
        }

        self._ep_raw = {
            'mass': mass_frac, 'latency': latency, 'cfm': cfm, 'erp': erp,
        }
        self._episode_return = 0.0

        rospy.loginfo(
            'ADR ep%d: mass=%.3f lat=%.1fms cfm=%.5f erp=%.3f | '
            'widths mass=%.3f lat=%.1fms cfm=%.5f erp=%.3f',
            self._episode_count + 1,
            mass_frac, latency * 1000, cfm, erp,
            self._curr_ranges['mass'][1]    - self._curr_ranges['mass'][0],
            (self._curr_ranges['latency'][1] - self._curr_ranges['latency'][0]) * 1000,
            self._curr_ranges['cfm'][1]     - self._curr_ranges['cfm'][0],
            self._curr_ranges['erp'][1]     - self._curr_ranges['erp'][0],
        )

        # Bypass RosPugEnvDR.reset() to avoid re-sampling from fixed constructor ranges
        obs, _ = RosPugEnv.reset(self, seed=seed, options=options)
        return obs, {
            'dr_params':  self._dr_params,
            'adr_ranges': {k: list(v) for k, v in self._curr_ranges.items()},
        }

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        self._episode_return += reward
        # Include current ranges in every step so the callback can log them
        info['adr_ranges'] = {k: list(v) for k, v in self._curr_ranges.items()}

        if terminated or truncated:
            info['adr_state'] = self._end_episode()

        return obs, reward, terminated, truncated, info

    def _end_episode(self) -> Dict[str, Any]:
        """Record episode outcome and optionally trigger ADR range expansion."""
        self._buffer.append({'params': dict(self._ep_raw), 'return': self._episode_return})
        self._episode_count += 1
        if (self._episode_count % self._update_interval == 0
                and len(self._buffer) >= self._update_interval):
            self._update_ranges()
            self._update_count += 1
        return self._get_adr_state()

    # ------------------------------------------------------------------
    # SW-ADR sensitivity computation and range expansion
    # ------------------------------------------------------------------

    def _update_ranges(self) -> None:
        """
        Core SW-ADR step: compute per-dimension sensitivity, normalise to weights,
        expand each range proportionally to its sensitivity weight.
        """
        sensitivities: Dict[str, float] = {}

        for d in self._DIMS:
            lo, hi  = self._curr_ranges[d]
            bw      = self._boundary_frac * (hi - lo)

            boundary_rets = [
                ep['return'] for ep in self._buffer
                if ep['params'][d] < lo + bw or ep['params'][d] > hi - bw
            ]
            interior_rets = [
                ep['return'] for ep in self._buffer
                if lo + bw <= ep['params'][d] <= hi - bw
            ]

            if len(boundary_rets) < self._min_boundary_eps:
                sensitivities[d] = 0.0
                continue

            var_b = float(np.var(boundary_rets))
            # Use boundary variance as baseline when interior is too small to estimate
            var_i = float(np.var(interior_rets)) if len(interior_rets) >= 2 else var_b
            sensitivities[d] = max(0.0, var_b - var_i)

        self._last_sensitivities = dict(sensitivities)

        # Normalise; fall back to uniform weights if no dimension shows brittleness
        total = sum(sensitivities.values())
        if total > 0:
            weights = {d: sensitivities[d] / total for d in self._DIMS}
        else:
            weights = {d: 0.25 for d in self._DIMS}

        self._last_weights = dict(weights)

        # Expand each dimension proportionally to its sensitivity weight
        for d in self._DIMS:
            half_step = self._EXPAND_BUDGET[d] * weights[d]
            self._curr_ranges[d][0] = max(
                self._curr_ranges[d][0] - half_step,
                self._MAX_RANGES[d][0],
            )
            self._curr_ranges[d][1] = min(
                self._curr_ranges[d][1] + half_step,
                self._MAX_RANGES[d][1],
            )

        rospy.loginfo(
            'ADR update #%d | sensitivities (×1e3): mass=%.2f lat=%.2f cfm=%.2f erp=%.2f | '
            'weights: %.2f/%.2f/%.2f/%.2f | '
            'new widths: mass=%.3f lat=%.1fms cfm=%.5f erp=%.3f',
            self._update_count + 1,
            *(sensitivities[d] * 1e3 for d in self._DIMS),
            *(weights[d] for d in self._DIMS),
            self._curr_ranges['mass'][1]    - self._curr_ranges['mass'][0],
            (self._curr_ranges['latency'][1] - self._curr_ranges['latency'][0]) * 1000,
            self._curr_ranges['cfm'][1]     - self._curr_ranges['cfm'][0],
            self._curr_ranges['erp'][1]     - self._curr_ranges['erp'][0],
        )

    def _get_adr_state(self) -> Dict[str, Any]:
        return {
            'ranges':        {k: list(v) for k, v in self._curr_ranges.items()},
            'sensitivities': dict(self._last_sensitivities),
            'weights':       dict(self._last_weights),
            'update_count':  self._update_count,
            'episode_count': self._episode_count,
        }
