---
mode: agent
description: ROSPug Sim-to-Real RL project — full context for continuing development
---

# ROSPug Sim-to-Real RL Project — Copilot Context

## Research Goal
Build a **lightweight sim-to-real transfer pipeline with onboard domain randomization** for the
Hiwonder ROSPug quadruped robot. The key research contribution:

- Train a reinforcement-learning gait policy in Gazebo across randomized friction / mass /
  servo-latency parameters
- Auto-tune the randomization range to match the real robot using a small number of real-world
  walking trials (a "reality gap estimator")
- Run the entire training pipeline on **Jetson Nano-class compute** (not a workstation GPU)

Minimum viable experiment: fixed-gait baseline vs. RL policy recovery from a push/slope on
terrain not seen during training.

---

## Robot Platform
| Property | Value |
|---|---|
| Robot | Hiwonder ROSPug (12-DOF quadruped, 4 legs × 3 serial bus servos) |
| Onboard computer | NVIDIA Jetson Nano (Ubuntu 18.04, JetPack 4.x) |
| ROS version | **ROS 1 Melodic** — catkin workspace, Python 3.6 |
| LiDAR | LDROBOT LD19 or MS200 (2D) |
| Camera | CSI (Raspberry-Pi connector on Jetson Nano) |
| Controller board | Hiwonder STM32 via CH342 USB-UART at `/dev/rrc`, 500 000 baud |

---

## Source Repositories (analysed)
| Repo | Purpose |
|---|---|
| `https://github.com/Hiwonder-docs/ROSPug` | Sphinx documentation only (ReadTheDocs). No installable ROS code. Contains ZIPs in `source/_static/source_code/`. |
| **`https://github.com/Hiwonder/ROSpug` branch `Jetson_nano_ros1`** | **← USE THIS. Full catkin workspace with 10 ROS packages.** |

### ROSpug package map
```
src/
├── pug_bringup/           launch entry points (base.launch)
├── pug_description/       URDF + Gazebo sim ← KEY for RL training
├── pug_driver/
│   ├── pug_control/       gait node: trot/amble/walk, velocity commands
│   ├── pug_sdk/           pure-Python IK/FK kinematics + PID + IMU fusion
│   └── ros_robot_controller/  STM32 serial driver (hardware only)
├── pug_slam/              GMapping launch/config
├── pug_navigation/        move_base + AMCL + TEB planner
├── pug_peripherals/       CSI camera via gscam  [EXCLUDED on x86]
├── pug_app/               TensorRT vision demos  [EXCLUDED on x86]
├── pug_example/           misc examples
└── pug_tutorial/          gait demos: demo01_trot_gait.py … demo05_trot_turn.py
```

### Simulation assets (pug_description)
- `urdf/pug.urdf.xacro` — full robot model
- `urdf/pug.gazebo.xacro` — Gazebo plugins: IMU, camera, **friction**, **inertia**
- `urdf/pug.transmission.xacro` — ros_control transmission interfaces
- `config/gazebo_control.yaml` — joint PID controllers (12 joints)
- `launch/gazebo.launch` — spawns robot + loads controllers
- `launch/sim_base.launch` — simulation bringup

**There is no existing RL code.** All gait control is classical (IK trajectories + PID).

---

## Development Environment
The developer machine runs **Ubuntu 22.04 + ROS 2 Humble** (used for another robot project).
ROSPug requires Ubuntu 18.04 + ROS Melodic, so a Docker container is used for PC-side development.

### Project directory: `~/rospug_research/`
```
Dockerfile              Ubuntu 18.04 (osrf/ros:melodic-desktop-full) + all deps + ROSPug built
docker-compose.yml      Service definition with X11 display forwarding for Gazebo GUI
entrypoint.sh           Sources /opt/ros/melodic/setup.bash + catkin_ws/devel/setup.bash
install_docker.sh       One-time Docker CE + Compose plugin installer for Ubuntu 22.04
run_gazebo.sh           xhost + docker compose run → roslaunch pug_description gazebo.launch
shell.sh                xhost + docker compose run → interactive bash shell
```

### Dockerfile highlights
- Base: `osrf/ros:melodic-desktop-full` (includes Gazebo 9, RViz, ros-control)
- Extra apt: `ros-melodic-navigation`, `gmapping`, `teb-local-planner`, `robot-localization`, etc.
- Extra pip: `pyserial`, `numpy`
- `pug_app/CATKIN_IGNORE` + `pug_peripherals/CATKIN_IGNORE` exclude ARM/Jetson-only packages
- `catkin_make -DCMAKE_BUILD_TYPE=Release` runs at build time

### docker-compose.yml highlights
- `network_mode: host` + `ipc: host` for clean ROS topic discovery
- `/tmp/.X11-unix` volume mount for Gazebo/RViz GUI via XWayland
- `DISPLAY`, `QT_X11_NO_MITSHM=1`, `LIBGL_ALWAYS_INDIRECT=0` set for GUI
- Volume `./rl:/root/catkin_ws/src/rl_rospug` is **commented out** — uncomment when RL packages exist

---

## Current Status
- [x] Both GitHub repos analysed and correct repo identified
- [x] All ROS + Python prerequisites documented
- [x] Docker dev environment designed and all files written to `~/rospug_research/`
- [ ] **Docker not yet installed** on the developer machine
- [ ] Docker image not yet built
- [ ] Gazebo simulation not yet verified

---

## Immediate Next Steps (in order)

### 1. Install Docker (requires sudo — run in terminal)
```bash
cd ~/rospug_research
sudo bash install_docker.sh
newgrp docker
```

### 2. Build the Docker image (~10–15 min, needs internet)
```bash
cd ~/rospug_research
docker compose build
```

### 3. Verify Gazebo simulation
```bash
./run_gazebo.sh
# Gazebo 9 should open showing the ROSPug URDF in an empty world.
# Verify joint controllers are loaded: rostopic list | grep joint
```

### 4. Smoke-test gait in simulation
```bash
./shell.sh
# inside container:
roslaunch pug_description sim_base.launch &
rosrun pug_tutorial demo01_trot_gait.py
```

---

## RL Pipeline — Design (not yet implemented)

### Phase 1: Domain Randomization Wrapper
Create a new ROS package `rl_rospug` (in `~/rospug_research/rl/`) that:
- Wraps `pug_description/gazebo.launch` with a Python `DomainRandomizer` class
- At episode reset, applies randomized parameters via Gazebo's `set_model_properties` service:
  - Friction: `mu1`, `mu2` on foot collision links (range: 0.3–1.2)
  - Link mass: ±20% of nominal values (affects inertia)
  - Joint PD gains: ±15% on `gazebo_control.yaml` kp/kd values
  - Action delay: inject N-step latency buffer in the action pipeline (0–3 steps)

### Phase 2: RL Environment (gym-compatible)
- Implement `PugGymEnv(gym.Env)` wrapping ROS topics:
  - **Observation**: joint positions (12) + joint velocities (12) + IMU orientation (4 quat)
    + base linear velocity (3) + base angular velocity (3) = 34-dim state
  - **Action**: 12 target joint positions (Δ from nominal stance, clipped to ±0.3 rad)
  - **Reward**: forward velocity − 0.1×control_cost − 5×fall_penalty
  - **Episode end**: base height < 0.05 m (fallen) or t > 10 s

### Phase 3: Policy Training (on PC, inside container)
- Algorithm: **PPO** (stable-baselines3) — low memory, works on CPU for small obs/action spaces
- Network: 2-layer MLP, 128 hidden units (fits Jetson Nano RAM for inference)
- Training: ~2M steps in randomized Gazebo sim

### Phase 4: Reality Gap Estimator
- Deploy policy to Jetson Nano; collect 5–10 real walking trials
- Record state trajectories; compare to sim rollouts
- Use Bayesian optimization (scikit-optimize) to find the randomization range
  that minimizes KL divergence between real and sim state distributions

### Phase 5: Evaluation
- Compare fixed-gait baseline vs. RL policy on unseen terrain (ramp, gravel, carpet)
- Metric: distance traveled before fall; disturbance recovery time

---

## Key Technical Constraints
- Jetson Nano: 4 GB RAM, 128-core Maxwell GPU → **inference only** after training
- Training must complete on PC CPU in reasonable time → keep policy network small
- ROS Melodic (Python 3.6): no f-strings with `=` format specifier; use `gym==0.21.0`
  (last version supporting Python 3.6); stable-baselines3==1.6.2
- `opencv-python` must NOT be pip-installed on Jetson — use JetPack OpenCV (≥ 4.5.4 needed
  for `cv2.FaceDetectorYN`; only matters for `pug_app` vision demos)
- Udev rule needed on real Jetson: `rosrun ros_robot_controller create_udev_rules.sh`
  (creates `/dev/rrc` symlink for CH342 USB-UART)

---

## ROS Topic/Service Reference (simulation)
| Topic / Service | Type | Notes |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | 12 joints, published by `robot_state_publisher` |
| `/pug/joint_group_position_controller/command` | `std_msgs/Float64MultiArray` | send 12 target positions |
| `/imu/data` | `sensor_msgs/Imu` | Gazebo IMU plugin on base_link |
| `/gazebo/set_model_configuration` | service | reset joint positions |
| `/gazebo/set_link_properties` | service | change inertia/friction at runtime |
| `/cmd_vel` | `geometry_msgs/Twist` | velocity command input to `pug_control` |
| `/odom` | `nav_msgs/Odometry` | ground truth in sim |
