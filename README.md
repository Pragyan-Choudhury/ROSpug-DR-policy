# ROSPug RL Gait Training

Reinforcement learning pipeline for training a quadruped gait policy on the **Hiwonder ROSPug** robot using **Proximal Policy Optimisation (PPO)** inside a Gazebo simulation.

The entire ROS + simulation environment runs inside a Docker container; only this repository is mounted as a volume. No ROS installation is needed on the host.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Docker container  (ROS 1 Melodic + Python 3.8)                  │
│                                                                  │
│  Gazebo 9 ── ROSPug URDF/controllers                            │
│      │ /pug/joint_states (100 Hz)                               │
│      │ /gazebo/model_states                                      │
│      ▼                                                           │
│  RosPugEnv  (gymnasium.Env)                                      │
│      obs : 26D  [12 joint pos + 12 joint vel + roll + pitch]    │
│      act : 12D  residual offsets on reference trot gait          │
│      rew : vx×3.0 + 0.5 − 10×fallen                            │
│      │                                                           │
│      ▼                                                           │
│  Stable-Baselines3 PPO  (MlpPolicy 256×256, GPU-accelerated)    │
│      checkpoints/ every 10 k steps                               │
│      TensorBoard logs/  → http://localhost:6006                  │
└─────────────────────────────────────────────────────────────────┘
        ▲  X11 socket          ▼  volume mount
   Host display (Gazebo GUI)   /home/<user>/rospug-rl-training/
```

---

## Current Progress

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Docker environment, ROSPug walking in Gazebo |
| 2 | ✅ | Gymnasium wrapper + sanity tests (20 eps, ~194 k steps/hr) |
| 3 | ✅ | Baseline PPO (Policy A) — 120 k+ steps saved |
| 4 | 🔄 | Continue / resume PPO training |
| 5 | ⬜ | Domain randomisation (friction, mass, servo latency) |
| 6 | ⬜ | Train Policy B on randomised environment |
| 7 | ⬜ | Head-to-head evaluation |
| 8 | ⬜ | Write-up |

---

## Host Prerequisites

### All systems
| Requirement | Version |
|---|---|
| Docker Engine | 24.0+ |
| Docker Compose plugin | 2.20+ |
| Git | any recent |
| Git LFS | 3.0+ |
| Free disk space | ≥ 15 GB (image is ~8–10 GB after build) |
| Display / X11 server | required for Gazebo GUI |

### GPU laptop (additional)
| Requirement | Notes |
|---|---|
| NVIDIA GPU | Any CUDA-capable card |
| NVIDIA driver | **≥ 520** (required for CUDA 11.8 wheels) |
| NVIDIA Container Toolkit | see install steps below |

---

## Setup: NVIDIA Container Toolkit (GPU Laptop)

Run these once on the host before building the image.

```bash
# 1. Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker daemon to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Verify (should print GPU info)
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Clone & Build

```bash
# 1. Clone the repo (Git LFS required for checkpoint files)
git clone https://github.com/<your-username>/rospug-rl-training.git
cd rospug-rl-training

# 2. Pull LFS objects (checkpoint .zip files)
git lfs pull

# 3. Allow GUI containers to use your display
xhost +local:docker

# 4. Build the Docker image (~20 min on first run)
#    This pulls ROS Melodic, clones ROSPug from Hiwonder's GitHub,
#    builds the catkin workspace, and installs the RL stack + CUDA PyTorch.
docker compose build
```

---

## Running

### Open a shell in the container
```bash
bash shell.sh
# or equivalently:
# xhost +local:docker && docker compose run --rm rospug bash
```

### Verify GPU access (inside the container)
```bash
python3.8 -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Launch Gazebo (terminal 1, inside container)
```bash
roslaunch pug_description gazebo.launch
```

### Sanity-test the environment (terminal 2, inside container)
```bash
cd /root/rospug_research
python3.8 scripts/test_env_random.py
# Expected: ≥50% episodes fall, all 20 complete without crash
```

### Train from scratch (terminal 2, inside container)
```bash
cd /root/rospug_research
python3.8 scripts/train_ppo.py
# Saves checkpoints every 10 k steps → checkpoints/
# TensorBoard logs   → logs/ppo_rospug/
```

### Resume from a checkpoint
```bash
python3.8 scripts/train_ppo.py \
    --resume checkpoints/ppo_rospug_120000_steps.zip \
    --timesteps 380000      # adds 380 k more steps (total 500 k)
```

### Evaluate a policy
```bash
python3.8 scripts/evaluate_policy.py \
    --model checkpoints/ppo_rospug_final.zip \
    --episodes 20
# Pass criteria: mean displacement ≥ 1.0 m AND fall rate < 50%
```

### Monitor training with TensorBoard
```bash
# Inside container (host-network mode → accessible at http://localhost:6006)
tensorboard --logdir logs/ppo_rospug/ --port 6006 --bind_all
```

---

## File Layout

```
rospug-rl-training/
├── Dockerfile                # ROS Melodic + Python 3.8 + SB3 + CUDA PyTorch
├── docker-compose.yml        # X11 forwarding, GPU passthrough, volume mount
├── entrypoint.sh             # Sources ROS + catkin env; fixes PYTHONPATH
├── shell.sh                  # Convenience: xhost + docker compose run bash
├── run_gazebo.sh             # Convenience: launches Gazebo inside container
├── install_docker.sh         # One-time Docker CE installer (Ubuntu 22.04)
├── requirements_rl.txt       # Python RL deps (without explicit torch — handled in Dockerfile)
├── rl_env/
│   ├── __init__.py
│   └── rospug_env.py         # gymnasium.Env wrapper over Gazebo/ROS
├── scripts/
│   ├── train_ppo.py          # PPO training entry point
│   ├── evaluate_policy.py    # Policy evaluation + metrics
│   └── test_env_random.py    # 20-episode random-action sanity test
├── sim_gait_controller_v3.py # Reference trot baseline (non-RL)
├── checkpoints/              # Saved PPO checkpoints (tracked with Git LFS)
├── logs/                     # TensorBoard event files
├── PROGRESS.md               # Detailed step-by-step progress log
└── ROSPug_2Week_Roadmap.md   # 2-week sprint plan
```

> **Note:** The ROSPug ROS workspace is cloned **inside the Docker image** from
> [Hiwonder/ROSpug (Jetson_nano_ros1)](https://github.com/Hiwonder/ROSpug/tree/Jetson_nano_ros1)
> during `docker compose build`. The `/home/arc09/ROSPug` directory on the host is
> **not** used — you do not need it on a fresh machine.

---

## GPU vs CPU Training

The Dockerfile installs PyTorch with **CUDA 11.8** wheels. Stable-Baselines3 automatically uses the GPU for policy network updates when `torch.cuda.is_available()` returns `True`.

- **GPU host** (NVIDIA Container Toolkit installed): PPO batch updates run on GPU → faster.
- **CPU-only host** (no NVIDIA toolkit): CUDA wheels load but fall back to CPU — training still works, just slower.

The primary training bottleneck is **Gazebo simulation speed** (real-time physics), not neural network compute, so GPU benefit is most visible during large batch updates (`n_steps=2048`, `batch_size=256`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot connect to X server` | Run `xhost +local:docker` on the host before starting the container |
| `GPU not available inside container` | Confirm `nvidia-ctk` is installed and Docker daemon restarted; run `nvidia-smi` on host |
| `protobuf` import errors | Check `protobuf>=3.20,<4.0` — version 4.x breaks TensorBoard |
| `ModuleNotFoundError: rospkg` | The `PYTHONPATH` fix in `entrypoint.sh` must run; use `bash shell.sh`, not `docker exec` directly |
| `catkin_make` errors during build | Usually a network issue pulling rosdep keys — re-run `docker compose build` |
| TensorBoard shows no data | Wait for the first `rollout/` log entry (~2048 steps); then refresh |
| Gazebo freezes / very slow | Disable physics via `/gazebo/pause_physics` service while not training |

---

## References

- [Hiwonder ROSPug](https://github.com/Hiwonder/ROSpug) — robot platform
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- [Gymnasium](https://gymnasium.farama.org/) — RL environment API
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
