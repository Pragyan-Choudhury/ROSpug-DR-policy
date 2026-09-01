#!/usr/bin/env python3.8
"""
train_ppo_c.py — Train Policy C (Sensitivity-Weighted ADR) on RosPugEnvADR (Step 6).

Policy C uses identical PPO hyperparameters to Policy A and B but trains on
RosPugEnvADR, which adaptively expands the most brittle physics dimension fastest.

Novel contribution: per-parameter return variance at range boundaries directs
differential curriculum expansion — the policy is automatically pushed hardest
where it is weakest.

TensorBoard (port 6008 to avoid clash with A/6006 and B/6007):
  python3.8 -m tensorboard.main --logdir /root/rospug_research/logs --port 6008

Key ADR TensorBoard metrics to watch (under the 'adr/' prefix):
  adr/range_mass_lo, adr/range_mass_hi       — body mass range evolution
  adr/range_latency_lo, adr/range_latency_hi — servo latency range evolution
  adr/range_cfm_lo, adr/range_cfm_hi         — ODE compliance range evolution
  adr/range_erp_lo, adr/range_erp_hi         — ODE erp range evolution
  adr/sensitivity_mass  etc.                  — per-parameter brittleness signal
  adr/weight_mass       etc.                  — normalised expansion budget share

Novel result to look for: at least one 'range_*_hi' curve has a steeper slope
than the others, confirming differential (sensitivity-weighted) expansion.

Checkpoints:
  checkpoints/ppo_rospug_c_10000_steps.zip ...
  checkpoints/policy_C_500k.zip  (on completion)
  checkpoints/policy_C_{N}steps.zip  (on interrupt)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env_adr import RosPugEnvADR

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

_ROOT          = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHECKPOINT_DIR = os.path.join(_ROOT, 'checkpoints')
LOG_DIR        = os.path.join(_ROOT, 'logs', 'ppo_rospug_c')

# Identical to Policy A and B — only the environment differs
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


class ADRLoggingCallback(BaseCallback):
    """
    Reads adr_state from episode-end info dicts and forwards ADR metrics
    (range bounds and sensitivity weights) to TensorBoard.
    """

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'adr_state' not in info:
                continue
            state = info['adr_state']
            ranges = state['ranges']
            senss  = state['sensitivities']
            wts    = state['weights']

            for dim in ('mass', 'latency', 'cfm', 'erp'):
                self.logger.record(f'adr/range_{dim}_lo', ranges[dim][0])
                self.logger.record(f'adr/range_{dim}_hi', ranges[dim][1])
                self.logger.record(f'adr/sensitivity_{dim}', senss[dim])
                self.logger.record(f'adr/weight_{dim}', wts[dim])

            self.logger.record('adr/update_count',  state['update_count'])
            self.logger.record('adr/episode_count', state['episode_count'])

        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Train Policy C (SW-ADR) on RosPugEnvADR. '
            'Gazebo must be running with Play pressed before launching.'
        )
    )
    p.add_argument(
        '--timesteps', type=int, default=500_000,
        help='Total environment steps (default: 500,000)',
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
    print('RosPugEnvADR  PPO Training  —  Policy C  (Step 6)')
    print(f'  Novel method    : Sensitivity-Weighted ADR')
    print(f'  Total timesteps : {args.timesteps:,}')
    print(f'  Resume from     : {args.resume or "scratch"}')
    print(f'  Checkpoints dir : {CHECKPOINT_DIR}')
    print(f'  TensorBoard dir : {LOG_DIR}  (port 6008)')
    print('=' * 60)
    print()

    env = RosPugEnvADR()

    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = 'ppo_rospug_c',
        verbose     = 2,
    )
    adr_cb = ADRLoggingCallback(verbose=0)

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

    # SB3 adds model.num_timesteps to total_timesteps when reset_num_timesteps=False,
    # so pass only the remaining steps to land at the correct absolute target.
    remaining = max(0, args.timesteps - model.num_timesteps)
    print(f'[INFO] Steps: {model.num_timesteps:,} done, {remaining:,} remaining (target {args.timesteps:,})')

    completed = False
    try:
        model.learn(
            total_timesteps     = remaining,
            callback            = [checkpoint_cb, adr_cb],
            reset_num_timesteps = False,
        )
        completed = True
    finally:
        actual = model.num_timesteps
        name   = 'policy_C_500k' if completed else f'policy_C_{actual}steps'
        final_path = os.path.join(CHECKPOINT_DIR, name)
        model.save(final_path)
        status = 'complete' if completed else f'interrupted at {actual:,} steps'
        print(f'\n[INFO] Policy C checkpoint ({status})  →  {final_path}.zip')
        env.close()


if __name__ == '__main__':
    main()
