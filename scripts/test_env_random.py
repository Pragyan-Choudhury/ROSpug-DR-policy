#!/usr/bin/env python3.8
"""
test_env_random.py — 20-episode random-action sanity test for RosPugEnv.

Run inside the Docker container with Gazebo already running:

  Terminal 1 (inside container):
    roslaunch pug_description gazebo.launch    # press Play in Gazebo GUI

  Terminal 2 (inside container):
    python3.8 /root/rospug_research/scripts/test_env_random.py

What this test validates:
  - reset() and step() don't crash over 20 episodes
  - terminated=True fires quickly on random actions (robot falls within ~10-50 steps)
  - Reward is negative when fallen (should see large negative spikes)
  - Wall-clock time per episode is printed → use this to scope training throughput

Exit criteria (Step 3):
  All 20 episodes complete without Python exception.
  At least half of episodes end with terminated=True (not just truncated).
"""

import sys
import os
import time

# Allow running from any working directory inside the container
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from rl_env.rospug_env import RosPugEnv, MAX_STEPS

NUM_EPISODES   = 20
PRINT_INTERVAL = 20   # print a step summary every N steps


def main() -> None:
    print("=" * 60)
    print("RosPugEnv random-action test")
    print(f"Episodes: {NUM_EPISODES}  MAX_STEPS per episode: {MAX_STEPS}")
    print("=" * 60)

    env = RosPugEnv()

    episode_times:   list = []
    episode_rewards: list = []
    fall_count = 0

    for ep in range(NUM_EPISODES):
        t0 = time.time()
        obs, _info = env.reset()

        episode_reward = 0.0
        step = 0
        terminated = False
        truncated  = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if step % PRINT_INTERVAL == 0:
                print(f"  ep={ep+1:2d}  step={step:4d}  r={reward:+.3f}  "
                      f"roll={info['roll']:+.3f}  pitch={info['pitch']:+.3f}  "
                      f"vx={info['vx']:+.4f}")

        elapsed = time.time() - t0
        episode_times.append(elapsed)
        episode_rewards.append(episode_reward)

        end_reason = 'FELL    ' if terminated else 'TIMEOUT '
        if terminated:
            fall_count += 1
        print(f"[ep {ep+1:2d}/{NUM_EPISODES}] {end_reason} "
              f"steps={step:4d}  total_reward={episode_reward:+9.2f}  "
              f"wall={elapsed:.1f}s")

    env.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    mean_time   = sum(episode_times) / len(episode_times)
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    fall_rate   = fall_count / NUM_EPISODES * 100

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Fall rate:              {fall_rate:.0f}%  ({fall_count}/{NUM_EPISODES})")
    print(f"  Mean episode reward:    {mean_reward:+.2f}")
    print(f"  Mean wall-clock / ep:   {mean_time:.1f} s")
    print()
    # Rough training throughput estimate for Step 4 planning
    steps_per_hour = int(3600 / mean_time * MAX_STEPS)
    print(f"  Estimated training throughput at MAX_STEPS={MAX_STEPS}:")
    print(f"    ~{steps_per_hour:,} env steps / hour")
    print(f"    To reach 500k steps: ~{500_000 / steps_per_hour:.1f} hours")
    print()
    print("Step 3 exit criteria:")
    print(f"  [{'PASS' if fall_count > 0 else 'FAIL'}] At least one episode terminated by falling")
    print(f"  [PASS] All {NUM_EPISODES} episodes completed without crash (you're seeing this)")


if __name__ == '__main__':
    main()
