#!/usr/bin/env python3.8
"""
verify_adr.py — Fast functional verification of RosPugEnvADR (Step 6).

Runs 25 short episodes and checks that SW-ADR machinery fires correctly:
  • PASS 1 (ep 10) — at least one range bound has changed after the first update.
  • PASS 2 (ep 20) — after the second update, sensitivity values differ across
                     dimensions (confirming non-uniform weighting).

Does NOT train any policy — actions are random (zero-mean, small noise) to
keep the robot roughly upright and allow episodes to proceed at the gym level.
Requires Gazebo to be running with a clean ROSPug world loaded.

Usage (inside Docker):
    cd /root/rospug_research
    python3.8 scripts/verify_adr.py

Expected runtime: ~8-12 min (25 episodes × ~10 steps each at 50 Hz).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from rl_env.rospug_env_adr import RosPugEnvADR

# ── Verification parameters ────────────────────────────────────────────────────
N_EPISODES          = 25
STEPS_PER_EPISODE   = 10
UPDATE_INTERVAL     = 10   # fire first update after 10 episodes
BUFFER_SIZE         = 10
MIN_BOUNDARY_EPS    = 3    # lower than default so the buffer triggers reliably

FIRST_UPDATE_EP     = UPDATE_INTERVAL       # 10
SECOND_UPDATE_EP    = 2 * UPDATE_INTERVAL   # 20


def ranges_snapshot(env: RosPugEnvADR) -> dict:
    """Return a deep copy of current ranges."""
    return {k: list(v) for k, v in env._curr_ranges.items()}


def any_range_changed(before: dict, after: dict) -> bool:
    for dim in before:
        if not np.allclose(before[dim], after[dim], atol=1e-9):
            return True
    return False


def sensitivities_differ(env: RosPugEnvADR) -> bool:
    """True when at least two sensitivity values are not the same."""
    vals = list(env._last_sensitivities.values())
    return not all(abs(v - vals[0]) < 1e-9 for v in vals)


def main() -> None:
    print('=' * 60)
    print('verify_adr.py  —  RosPugEnvADR functional verification')
    print(f'  {N_EPISODES} episodes × {STEPS_PER_EPISODE} steps')
    print(f'  update_interval={UPDATE_INTERVAL}  buffer_size={BUFFER_SIZE}')
    print('=' * 60)

    env = RosPugEnvADR(
        update_interval  = UPDATE_INTERVAL,
        buffer_size      = BUFFER_SIZE,
        min_boundary_eps = MIN_BOUNDARY_EPS,
    )

    initial_ranges = None
    ranges_after_1 = None
    pass1 = None   # bool
    pass2 = None   # bool
    failures: list = []

    try:
        for ep in range(1, N_EPISODES + 1):
            obs, info = env.reset()

            if ep == 1:
                initial_ranges = ranges_snapshot(env)
                print('\n[INIT] Starting ranges:')
                _print_ranges(initial_ranges)

            ep_return = 0.0
            terminated = truncated = False
            for _ in range(STEPS_PER_EPISODE):
                action = env.action_space.sample() * 0.05
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                if terminated or truncated:
                    break
            # Finalize episode for ADR buffer if it didn't naturally end
            if not (terminated or truncated):
                env._end_episode()

            print(f'  ep {ep:3d}  return={ep_return:+.3f}  '
                  f'update_count={env._update_count}  '
                  f'buffer={len(env._buffer)}', flush=True)

            # ── PASS 1: after first ADR update ────────────────────────────────
            if ep == FIRST_UPDATE_EP:
                if env._update_count < 1:
                    failures.append(
                        f'FAIL 1: no update fired after {FIRST_UPDATE_EP} episodes '
                        f'(update_count={env._update_count})')
                    pass1 = False
                else:
                    ranges_after_1 = ranges_snapshot(env)
                    pass1 = any_range_changed(initial_ranges, ranges_after_1)
                    print(f'\n[CHECK 1] Ranges after first update:')
                    _print_ranges(ranges_after_1)
                    print(f'  Sensitivities: {env._last_sensitivities}')
                    print(f'  Weights:       {env._last_weights}')
                    if not pass1:
                        failures.append(
                            'FAIL 1: no range bound changed after first ADR update')

            # ── PASS 2: after second ADR update ───────────────────────────────
            if ep == SECOND_UPDATE_EP:
                if env._update_count < 2:
                    failures.append(
                        f'FAIL 2: second update not fired after {SECOND_UPDATE_EP} episodes '
                        f'(update_count={env._update_count})')
                    pass2 = False
                else:
                    ranges_after_2 = ranges_snapshot(env)
                    pass2 = sensitivities_differ(env)
                    print(f'\n[CHECK 2] Ranges after second update:')
                    _print_ranges(ranges_after_2)
                    print(f'  Sensitivities: {env._last_sensitivities}')
                    print(f'  Weights:       {env._last_weights}')
                    if not pass2:
                        failures.append(
                            'FAIL 2: all sensitivity values identical — SW-ADR not differentiating')

    finally:
        env.close()

    print('\n' + '=' * 60)
    if pass1 is None:
        failures.append(f'FAIL 1: first update never reached (only ran {N_EPISODES} episodes)')
    if pass2 is None:
        failures.append(f'FAIL 2: second update never reached (only ran {N_EPISODES} episodes)')

    if not failures:
        print('[PASS]  Both ADR checks passed.')
        print('        SW-ADR ranges are evolving and sensitivities differ across dimensions.')
    else:
        print('[FAIL]  One or more checks failed:')
        for msg in failures:
            print(f'  • {msg}')

    # Print final state regardless of pass/fail
    print('\nFinal ADR state:')
    print(f'  episode_count = {env._episode_count}')
    print(f'  update_count  = {env._update_count}')
    print(f'  buffer length = {len(env._buffer)}')
    print(f'  sensitivities = {env._last_sensitivities}')
    print(f'  weights       = {env._last_weights}')
    print('  final ranges:')
    _print_ranges(env._curr_ranges)

    sys.exit(0 if not failures else 1)


def _print_ranges(ranges: dict) -> None:
    for dim, (lo, hi) in ranges.items():
        width = hi - lo
        if dim == 'latency':
            print(f'    {dim:8s}  [{lo*1000:6.2f}, {hi*1000:6.2f}] ms  (width={width*1000:.2f} ms)')
        else:
            print(f'    {dim:8s}  [{lo:.5f}, {hi:.5f}]  (width={width:.5f})')


if __name__ == '__main__':
    main()
