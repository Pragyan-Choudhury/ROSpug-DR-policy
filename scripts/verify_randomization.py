#!/usr/bin/env python3.8
"""
verify_randomization.py — Sanity-check that RosPugEnvDR samples different
physics parameters each episode.

Run inside Docker with Gazebo already running (Play pressed):

  python3.8 /root/rospug_research/scripts/verify_randomization.py

Expected output (values will differ each run):
  Episode  1/10  mass_frac=0.923  latency= 4.7ms  cfm=0.000231  erp=0.612
  Episode  2/10  mass_frac=1.082  latency= 8.1ms  cfm=0.000044  erp=0.441
  ...
  [PASS] 10/10 episodes have distinct mass_frac values.

Run this before starting a long training run to confirm the DR mechanism
works end-to-end; failures here indicate a service or import problem rather
than a training problem.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env_dr import RosPugEnvDR

EPISODES          = 10
STEPS_PER_EPISODE = 10   # short — we only need to verify reset() params, not full episodes


def main() -> None:
    env = RosPugEnvDR()
    print(f'\nVerifying RosPugEnvDR over {EPISODES} episodes '
          f'({STEPS_PER_EPISODE} steps each) ...\n')

    mass_fracs = []

    for ep in range(1, EPISODES + 1):
        obs, info = env.reset()
        dr = info['dr_params']
        mass_fracs.append(dr['mass_frac'])

        print(
            f"  Episode {ep:2d}/{EPISODES}"
            f"  mass_frac={dr['mass_frac']:.3f}"
            f"  latency={dr['latency_ms']:5.1f}ms"
            f"  cfm={dr['cfm']:.6f}"
            f"  erp={dr['erp']:.3f}"
        )

        for _ in range(STEPS_PER_EPISODE):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()

    # All mass_frac values rounded to 3 d.p. should be distinct
    unique_count = len(set(round(m, 3) for m in mass_fracs))
    if unique_count == EPISODES:
        print(f'\n[PASS] {unique_count}/{EPISODES} episodes have distinct mass_frac values.')
        print('       Domain randomization is active. Safe to start train_ppo_b.py.')
    else:
        print(f'\n[WARN] Only {unique_count}/{EPISODES} distinct mass_frac values.')
        print('       Check that /gazebo/set_link_properties service is responding.')


if __name__ == '__main__':
    main()
