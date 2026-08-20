#!/usr/bin/env python3.8
"""
evaluate_policy.py — Load a saved PPO checkpoint and evaluate it for N episodes.

Run inside the Docker container with Gazebo already running:

  Terminal 1 (inside container):
    roslaunch pug_description gazebo.launch    # press Play ▶ in Gazebo GUI

  Terminal 2 (inside container):
    python3.8 /root/rospug_research/scripts/evaluate_policy.py \\
        --checkpoint /root/rospug_research/checkpoints/ppo_rospug_100000_steps.zip

Output per episode:
  status (FELL / SURVIVED), x-displacement (m), total reward, episode length

Summary printed at end:
  mean x-displacement, fall rate, mean reward, mean steps

Step 4 exit criterion (printed at end):
  mean displacement >= 1.0 m  AND  fall rate < 50%
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env import RosPugEnv, DT

from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Evaluate a saved PPO policy on RosPugEnv. '
            'Gazebo must be running with Play pressed.'
        )
    )
    p.add_argument(
        '--checkpoint', required=True, metavar='PATH',
        help='Path to a PPO .zip checkpoint file',
    )
    p.add_argument(
        '--episodes', type=int, default=10,
        help='Number of evaluation episodes (default: 10)',
    )
    p.add_argument(
        '--stochastic', action='store_true',
        help='Use stochastic actions (default: deterministic)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic

    print('=' * 60)
    print('RosPugEnv Policy Evaluation  —  Step 4')
    print(f'  Checkpoint  : {args.checkpoint}')
    print(f'  Episodes    : {args.episodes}')
    print(f'  Mode        : {"deterministic" if deterministic else "stochastic"}')
    print('=' * 60)

    env   = RosPugEnv()
    model = PPO.load(args.checkpoint)

    results = []

    for ep in range(args.episodes):
        obs, _ = env.reset()

        ep_reward      = 0.0
        x_displacement = 0.0
        fell           = False
        steps          = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward      += reward
            x_displacement += info['vx'] * DT   # dead-reckoning from vx already in info
            steps          += 1

            if terminated:
                fell = True
            if terminated or truncated:
                break

        results.append(dict(
            displacement = x_displacement,
            fell         = fell,
            reward       = ep_reward,
            steps        = steps,
        ))

        status = 'FELL    ' if fell else 'SURVIVED'
        print(f'[ep {ep+1:2d}/{args.episodes}] {status}  '
              f'x={x_displacement:+6.2f} m  '
              f'reward={ep_reward:+8.2f}  steps={steps:4d}')

    env.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n          = len(results)
    mean_disp  = sum(r['displacement'] for r in results) / n
    fall_count = sum(1 for r in results if r['fell'])
    fall_rate  = fall_count / n * 100.0
    mean_rew   = sum(r['reward'] for r in results) / n
    mean_steps = sum(r['steps'] for r in results) / n

    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'  Mean x-displacement : {mean_disp:+.3f} m')
    print(f'  Fall rate           : {fall_rate:.0f}%  ({fall_count}/{n})')
    print(f'  Mean episode reward : {mean_rew:+.2f}')
    print(f'  Mean steps / ep     : {mean_steps:.0f}')
    print()

    passed_disp = mean_disp >= 1.0
    passed_fall = fall_rate < 50.0

    if passed_disp and passed_fall:
        print('Step 4 exit criterion: [PASS]  displacement >= 1.0 m  AND  fall rate < 50%')
    else:
        reasons = []
        if not passed_disp:
            reasons.append(f'displacement {mean_disp:+.2f} m  <  1.0 m')
        if not passed_fall:
            reasons.append(f'fall rate {fall_rate:.0f}%  >=  50%')
        print('Step 4 exit criterion: [NOT MET]')
        for r in reasons:
            print(f'    - {r}')

    print('=' * 60)


if __name__ == '__main__':
    main()
