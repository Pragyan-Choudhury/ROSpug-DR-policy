# ROSPug RL Project — Progress Tracker

## Overview

Training an RL walking policy for ROSPug in Gazebo with domain randomization.
**Core deliverable:** Policy A (fixed physics) vs Policy B (randomised physics) — show B is more robust on held-out conditions.

---

## Step 1 — Environment Setup ✅
**Goal:** ROS1 + Gazebo + Python RL stack installed and talking to each other.

**Completed:**
- Docker image built from `osrf/ros:melodic-desktop-full` (ROS 1 Melodic, Ubuntu 18.04)
- Gazebo 9 functional with `roslaunch pug_description gazebo.launch`
- `sim_gait_controller_v3.py` runs in second container shell, responds to `/cmd_vel`
- Python RL stack (`stable-baselines3`, `gymnasium`) identified; Python 3.8 upgrade planned for Step 3

**Key files:**
- `rospug_research/Dockerfile` — Docker build recipe
- `rospug_research/docker-compose.yml` — container config (host networking, X11 forwarding)
- `rospug_research/entrypoint.sh` — sources ROS Melodic + catkin devel setup

---

## Step 2 — ROSPug Walking in Gazebo ✅
**Goal:** ROSPug's simulated model loads in Gazebo and walks using its original, non-RL gait controller.

**Completed:**
- URDF spawns cleanly: `gazebo.launch` → `spawn_model -model pug`
- 12 effort controllers verified live: `/pug/{leg}_{joint}_position_controller/command`
- Joint state feedback confirmed at `/pug/joint_states` (100 Hz)
- `sim_gait_controller_v3.py` — joint-space trot with forward, backward, yaw, strafe
- Commands verified: `rostopic pub /cmd_vel geometry_msgs/Twist` → visible motion in Gazebo

**Key files:**
- `ROSPug/src/pug_description/launch/gazebo.launch` — simulation entry point
- `ROSPug/src/pug_description/config/gazebo_control.yaml` — 12 PID joint controllers
- `rospug_research/sim_gait_controller_v3.py` — reference gait: trot + yaw + strafe

**Findings (important for Step 3):**
- Stand pose = all 12 joints at 0.0 rad
- Gazebo model name in `/gazebo/model_states`: `"pug"`
- No IMU Gazebo plugin in URDF — body orientation must come from `/gazebo/model_states`
- Joint state topic is `/pug/joint_states` (remapped from `/joint_states`)

---

## Step 3 — Gym-style RL Environment ✅
**Goal:** `RosPugEnv` — a `gymnasium.Env` subclass that lets Stable-Baselines3 treat ROSPug-in-Gazebo like any other RL environment.

**Completed:** 2026-08-18

---

### What Was Built

| File | Purpose |
|------|---------|
| `rospug_research/rl_env/__init__.py` | Package marker; exports `RosPugEnv` |
| `rospug_research/rl_env/rospug_env.py` | Full `gymnasium.Env` implementation |
| `rospug_research/scripts/test_env_random.py` | 20-episode random-action sanity test |
| `rospug_research/requirements_rl.txt` | RL package pins |

### Docker changes
| File | Change |
|------|--------|
| `Dockerfile` | Python 3.8 via deadsnakes PPA; pip: gymnasium, stable-baselines3, tensorboard, rospkg, catkin-pkg, netifaces |
| `docker-compose.yml` | `:ro` → `:rw` mount; added `.Xauthority` for Gazebo GUI |
| `entrypoint.sh` | Prepends `python3/dist-packages` to PYTHONPATH |

### Bugs fixed during integration
| Error | Fix |
|-------|-----|
| `get-pip.py` requires Python ≥ 3.10 | Changed URL to `bootstrap.pypa.io/pip/3.8/get-pip.py` |
| `apt install python3-rospy` not in Melodic snapshot | Removed; installed `rospkg catkin-pkg` via pip instead |
| `import tf.transformations` → Python-2.7 C extension crash | Replaced with 6-line inline `_quat_to_rpy()` using `math` only |
| `ModuleNotFoundError: No module named 'netifaces'` | `python3.8 -m pip install netifaces` + added to Dockerfile |
| `docker exec` overwrites PYTHONPATH via `.bashrc` | Added PYTHONPATH export to `.bashrc` in Dockerfile |
| Gazebo GUI crash (`Authorization required`) | Added `.Xauthority` mount + `XAUTHORITY` env var to `docker-compose.yml` |

---

### Design Summary
| Aspect | Value |
|--------|-------|
| Action space | Box(−0.5, 0.5, shape=(12,)) — residual joint offsets from stand pose (all zeros) |
| Observation | Box(−∞, ∞, shape=(26,)) — 12 joint pos + 12 joint vel + body roll + pitch |
| Body orientation | Inline quaternion→RPY from `/gazebo/model_states` (no tf dependency) |
| Reward | `vx × 1.0 − 0.001 × Σ(joint_vel²) − 10.0 × fallen` |
| Termination | `|roll| > 0.7 rad` OR `|pitch| > 0.7 rad` |
| Truncation | 500 steps (10 s at 50 Hz) |

---

### Test Results (2026-08-18)

```
Episodes: 20  MAX_STEPS: 500
------------------------------------------------------------
Fall rate:              30%  (6/20)
Mean episode reward:    -529.74
Mean wall-clock / ep:   9.3 s
Estimated throughput:   ~194,490 env steps / hour
To reach 500k steps:    ~2.6 hours
------------------------------------------------------------
[PASS] At least one episode terminated by falling
[PASS] All 20 episodes completed without crash
```

**Observations for Step 4:**
- 70% of episodes hit MAX_STEPS (truncated) — the stand pose (all zeros) is stable even under random ±0.5 rad perturbations, which is good: PPO will start from a stable configuration
- 30% fall rate shows the termination condition is working correctly
- Mean reward −529.74 / 500 steps ≈ −1.06/step is dominated by energy penalty (expected with random actions)
- Training throughput ~194k steps/hour means 500k steps ≈ 2.6 hours — feasible in one session
- **Action space may need widening** to ±0.8–1.0 rad for PPO (±0.5 rad is hard to fall with, so falling penalty is weak signal)

---

## Step 4 — Train Baseline PPO Policy ⬜
**Goal:** Policy A — walks forward in Gazebo on fixed physics parameters.

**Start conditions (from Step 3):**
- Training throughput: ~194k steps/hour → budget **500k–1M steps** (3–5 hours)
- Stand pose is stable → PPO will learn from a good starting point
- Episode length: mostly 500 steps (robot doesn't fall) → reward signal comes primarily from `vx`

**Planned approach:**
1. Write `scripts/train_ppo.py` using Stable-Baselines3 PPO on `RosPugEnv`
2. Log to TensorBoard; checkpoint every 10k steps
3. Evaluate every 30 min by watching Gazebo for visible forward progress
4. Exit criterion: robot walks forward several body-lengths without falling in the training conditions

---

## Step 5 — Domain Randomisation + Train Policy B ⬜
**Goal:** Policy B — trained on randomised friction / mass / servo latency.

---

## Step 6 — Head-to-Head Evaluation ⬜
**Goal:** Quantitatively show Policy B generalises better on held-out physics conditions.

---

## Step 7 — Stretch: Real Hardware Test ⬜
**Goal:** Deploy Policy B to real ROSPug, compare against fixed-gait baseline.

---

## Step 8 — Write-Up ⬜

---

## Reference: Roadmap Original Document
`rospug_research/ROSPug_2Week_Roadmap.md`
