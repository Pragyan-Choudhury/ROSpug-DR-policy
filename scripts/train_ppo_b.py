#!/usr/bin/env python3.8
"""
train_ppo_b.py — Train Policy B (domain-randomized PPO) on RosPugEnvDR (Step 5).

Policy B trains with identical PPO hyperparameters to Policy A but on
RosPugEnvDR, which randomises:
  - body mass ±15%
  - servo latency 0–10 ms
  - ODE surface compliance (cfm/erp)
at the start of every episode.

Policy B starts from scratch (not from Policy A) so the two policies have
identical training budgets and the same random initialisation — the only
difference is the environment.

Prerequisites:
  Terminal 1: roslaunch pug_description gazebo.launch   # press Play ▶

Run training:
  python3.8 /root/rospug_research/scripts/train_ppo_b.py

TensorBoard (use port 6007 to avoid clash with Policy A logs on 6006):
  python3.8 -m tensorboard.main \\
      --logdir /root/rospug_research/logs --port 6007
  Browse http://localhost:6007

Checkpoint schedule:
  checkpoints/ppo_rospug_b_10000_steps.zip
  checkpoints/ppo_rospug_b_20000_steps.zip  ...
  checkpoints/policy_B_500k.zip   (final — always written on exit/interrupt)

Training time estimate:
  With up to 10 ms servo latency, each step takes ≤ 30 ms wall-clock
  (DT=20ms + latency≤10ms), so 500k steps ≈ 3–4 hours.

Decision tree if ep_rew_mean is still negative at 150k steps:
  Reduce randomization ranges, e.g.:
      env = RosPugEnvDR(mass_range=(0.90, 1.10), latency_range=(0.0, 0.005))
  This is a legitimate research finding — document the change in PROGRESS.md.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env_dr import RosPugEnvDR

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT          = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHECKPOINT_DIR = os.path.join(_ROOT, 'checkpoints')
LOG_DIR        = os.path.join(_ROOT, 'logs', 'ppo_rospug_b')

# ---------------------------------------------------------------------------
# PPO hyperparameters — identical to train_ppo.py (Policy A)
# ---------------------------------------------------------------------------
PPO_KWARGS = dict(
    policy        = 'MlpPolicy',
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    learning_rate = 3e-4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    policy_kwargs = dict(net_arch=[256, 256], log_std_init=-1.0),
    verbose       = 1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Train Policy B (domain-randomized PPO) on RosPugEnvDR. '
            'Gazebo must be running with Play pressed before launching.'
        )
    )
    p.add_argument(
        '--timesteps', type=int, default=500_000,
        help='Total environment steps to train (default: 500,000)',
    )
    p.add_argument(
        '--resume', type=str, default=None, metavar='CHECKPOINT',
        help='Path to a .zip checkpoint to continue training from',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print('=' * 60)
    print('RosPugEnvDR  PPO Training  —  Policy B  (Step 5)')
    print(f'  Total timesteps : {args.timesteps:,}')
    print(f'  Resume from     : {args.resume or "scratch"}')
    print(f'  DR ranges       : mass ±15%  latency 0-10ms  cfm/erp varied')
    print(f'  Checkpoints dir : {CHECKPOINT_DIR}')
    print(f'  TensorBoard dir : {LOG_DIR}')
    print('=' * 60)
    print()
    print('Run verify_randomization.py first to confirm DR is active.')
    print()

    env = RosPugEnvDR()

    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = 'ppo_rospug_b',
        verbose     = 2,   # SB3 2.x requires verbose>=2 to print checkpoint messages
    )

    if args.resume:
        print(f'[INFO] Loading checkpoint: {args.resume}')
        model = PPO.load(
            args.resume,
            env=env,
            tensorboard_log=LOG_DIR,
            custom_objects={'action_space': env.action_space},
        )
    else:
        model = PPO(env=env, tensorboard_log=LOG_DIR, **PPO_KWARGS)
    completed = False
    try:
        model.learn(
            total_timesteps     = args.timesteps,
            callback            = checkpoint_cb,
            reset_num_timesteps = (args.resume is None),
        )
        completed = True
    finally:
        actual = model.num_timesteps
        name   = 'policy_B_500k' if completed else f'policy_B_{actual}steps'
        final_path = os.path.join(CHECKPOINT_DIR, name)
        model.save(final_path)
        status = 'complete' if completed else f'interrupted at {actual:,} steps'
        print(f'\n[INFO] Policy B checkpoint ({status})  →  {final_path}.zip')
        env.close()


if __name__ == '__main__':
    main()
