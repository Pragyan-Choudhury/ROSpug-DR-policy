#!/usr/bin/env python3.8
"""
train_ppo.py — Train a PPO policy on RosPugEnv (Step 4).

Run inside the Docker container with Gazebo already running:

  Terminal 1 (inside container):
    roslaunch pug_description gazebo.launch    # press Play ▶ in the Gazebo GUI

  Terminal 2 (inside container):
    python3.8 /root/rospug_research/scripts/train_ppo.py

TensorBoard (third terminal inside container, or same terminal in background):
  python3.8 -m tensorboard.main --logdir /root/rospug_research/logs --port 6006
  Browse http://localhost:6006 from the host (host networking is active).

Checkpoint schedule:
  Saved every 10,000 steps to rospug_research/checkpoints/
    ppo_rospug_10000_steps.zip
    ppo_rospug_20000_steps.zip  ...
  Final model always written on clean exit (or keyboard interrupt):
    ppo_rospug_final.zip

Resume a previous run:
  python3.8 scripts/train_ppo.py \\
      --resume checkpoints/ppo_rospug_200000_steps.zip \\
      --timesteps 300000   # additional steps on top of resumed count

Key metrics to watch in TensorBoard:
  rollout/ep_rew_mean  — should rise above -529 (random-action baseline) within 50k steps
  rollout/ep_len_mean  — should increase as robot learns to stay upright
  train/approx_kl      — keep < 0.05; if higher, reduce learning rate to 1e-4
"""

import sys
import os
import argparse

# Allow running from any working directory inside the container
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env import RosPugEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------------------------
# Paths  (anchored relative to the rospug_research/ root)
# ---------------------------------------------------------------------------
_ROOT          = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHECKPOINT_DIR = os.path.join(_ROOT, 'checkpoints')
LOG_DIR        = os.path.join(_ROOT, 'logs', 'ppo_rospug')

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
PPO_KWARGS = dict(
    policy        = 'MlpPolicy',
    n_steps       = 2048,       # ~4 full 500-step episodes per rollout update
    batch_size    = 256,        # mini-batch size; must divide n_steps
    n_epochs      = 10,         # optimisation passes per rollout
    learning_rate = 3e-4,       # Adam lr — reduce to 1e-4 if approx_kl > 0.05
    gamma         = 0.99,       # discount; long-horizon (500-step, 10 s episodes)
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,       # entropy bonus — keeps exploration alive early
    vf_coef       = 0.5,
    policy_kwargs = dict(net_arch=[256, 256], log_std_init=-1.0),   # 2 hidden layers × 256 neurons
    verbose       = 1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Train PPO on RosPugEnv. '
            'Gazebo must be running with Play pressed before this script is launched.'
        )
    )
    p.add_argument(
        '--timesteps', type=int, default=500_000,
        help='Total environment steps to train (default: 500,000 ≈ 2.6 hours)',
    )
    p.add_argument(
        '--resume', type=str, default=None, metavar='CHECKPOINT',
        help='Path to a .zip checkpoint file to continue training from',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print('=' * 60)
    print('RosPugEnv PPO Training  —  Step 4')
    print(f'  Total timesteps : {args.timesteps:,}')
    print(f'  Resume from     : {args.resume or "scratch"}')
    print(f'  Checkpoints dir : {CHECKPOINT_DIR}')
    print(f'  TensorBoard dir : {LOG_DIR}')
    print('=' * 60)

    env = RosPugEnv()

    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = 'ppo_rospug',
        verbose     = 2,   # SB3 2.x requires verbose>=2 to print the save message
    )

    if args.resume:
        print(f'[INFO] Loading checkpoint: {args.resume}')
        # custom_objects overrides the saved action_space so the space check passes
        # when resuming from a checkpoint trained with a different action range.
        model = PPO.load(
            args.resume,
            env=env,
            tensorboard_log=LOG_DIR,
            custom_objects={'action_space': env.action_space},
        )
    else:
        model = PPO(env=env, tensorboard_log=LOG_DIR, **PPO_KWARGS)

    final_path = os.path.join(CHECKPOINT_DIR, 'ppo_rospug_final')

    try:
        model.learn(
            total_timesteps     = args.timesteps,
            callback            = checkpoint_cb,
            reset_num_timesteps = (args.resume is None),  # False = continue counter on resume
        )
    finally:
        model.save(final_path)
        print(f'\n[INFO] Final model saved  →  {final_path}.zip')
        env.close()


if __name__ == '__main__':
    main()
