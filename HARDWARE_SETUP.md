# ROSPug Real-Hardware Setup Reference

**Purpose:** Complete reference for connecting to, controlling, and verifying the physical ROSPug robot.
This document captures every step confirmed working in the pre-requisite phase for Step 8 (policy deployment).

---

## Table of Contents

1. [Hardware Network Setup](#1-hardware-network-setup)
2. [Robot Bringup (start ROS on robot)](#2-robot-bringup)
3. [Docker Container for Real Robot](#3-docker-container-for-real-robot)
4. [ROS Networking Architecture](#4-ros-networking-architecture)
5. [Teleop Test Module](#5-teleop-test-module)
6. [Observation Bridge Verification](#6-observation-bridge-verification)
7. [Quick Reconnect Reference](#7-quick-reconnect-reference-per-session)
8. [Infrastructure Files](#8-infrastructure-files)
9. [Step 8 Deployment Notes](#9-step-8-deployment-notes)

---

## 1. Hardware Network Setup

### Confirmed IPs

| Device | Interface | IP | Notes |
|--------|-----------|-----|-------|
| Laptop (host) | `enp2s0` | `192.168.123.100/24` | Ethernet, persistent via nmcli |
| Laptop (host) | `enp2s0` | `192.168.1.100/24` | Legacy, not used |
| ROSPug robot | `eth0` | `192.168.123.1/24` | Static, configured via nmcli |
| ROSPug robot | `wlan0` | `192.168.149.1` | WiFi AP gateway (backup access) |

### LAN Cable Connection (primary)

```bash
# Verify your laptop's Ethernet IP is set
ip addr show enp2s0 | grep "inet "
# Expected: inet 192.168.123.100/24

# If missing, restore it:
sudo ip addr add 192.168.123.100/24 dev enp2s0
```

### WiFi Backup Access (if Ethernet not available)

The robot broadcasts its own WiFi hotspot. Use this only if LAN cable isn't available.

```bash
# Disable campus WiFi auto-reconnect first
sudo nmcli connection modify i2web connection.autoconnect no

# Connect to robot's hotspot
nmcli device wifi connect "HW-6622B6A9" password "hiwonder"

# SSH via WiFi
ssh hiwonder@192.168.149.1   # password: hiwonder
```

After finishing, restore campus WiFi:
```bash
sudo nmcli connection modify i2web connection.autoconnect yes
nmcli device wifi connect "i2web"
```

### How the Ethernet Static IP Was Configured on the Robot

Done once via `nmcli` while SSH'd into the robot over WiFi:
```bash
sudo nmcli connection add \
    type ethernet ifname eth0 con-name "pug-eth" \
    ipv4.method manual ipv4.addresses "192.168.123.1/24" \
    connection.autoconnect yes
sudo nmcli connection up "pug-eth"
```
The robot remembers this across reboots.

---

## 2. Robot Bringup

### Robot OS and workspace

| Property | Value |
|----------|-------|
| OS | Ubuntu 18.04 (Jetson-based) |
| Shell | zsh |
| ROS version | Melodic |
| ROS workspace | `~/pug/` |
| Workspace setup file | `~/pug/devel/setup.zsh` |
| Default `ROS_MASTER_URI` | `http://localhost:11311` |
| Default `ROS_HOSTNAME` | `localhost` |

### Start bringup (run from laptop, before using container)

```bash
ssh hiwonder@192.168.123.1 \
  "nohup zsh -l -c 'roslaunch pug_bringup base.launch' > /tmp/bringup.log 2>&1 &"
# password: hiwonder
```

`zsh -l` (login shell) sources `~/.zshrc` automatically so no manual workspace sourcing is needed.

### Verify bringup started (wait 5 s)

```bash
sleep 5 && ssh hiwonder@192.168.123.1 \
  "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh && \
   rostopic list | grep joint_states"
# Expected: /joint_states
```

### Confirmed live ROS topics (with bringup running)

```
/imu
/imu_corrected
/joint_states
/pug_control/cmd_vel
/pug_control/gait
/pug_control/pose
/pug_control/velocity_move
/pug_control/web_controller
/ros_robot_controller/imu_raw
/ros_robot_controller/battery
```

---

## 3. Docker Container for Real Robot

### Container vs. simulation

| | Simulation container | Real-robot container |
|-|---------------------|----------------------|
| Config file | `docker-compose.yml` | `docker-compose.real.yml` |
| Container name | `rospug_dev` | `rospug_real` |
| ROS master | Internal (`localhost:11311`) | Robot over LAN (`192.168.123.1`) |
| GPU requirement | Yes (Gazebo) | No (removed) |

### Start the real-robot container

```bash
cd ~/rospug_research
HOST_ETH_IP=192.168.123.100 docker compose -f docker-compose.real.yml run --rm rospug bash
```

The `--rm` flag removes the container on exit (no stale state between sessions).

### Key environment variables inside the container

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROS_MASTER_URI` | `http://192.168.123.1:11311` | Points to robot's roscore |
| `ROS_IP` | `192.168.123.100` | Tells robot how to reach the container |
| `ROS_HOSTNAME` | `192.168.123.100` | Same as ROS_IP |
| `ROBOT_IP` | `192.168.123.1` | Used by scripts for SSH targets |
| `HOST_ETH_IP` | `192.168.123.100` | Your laptop's Ethernet IP |

### Why `network_mode: host`

The container shares the host's network stack. This means:
- Container can reach `192.168.123.1` directly (no Docker NAT)
- Container uses the host's `enp2s0` interface
- Required for ROS networking to work correctly

### Volume mount

`~/rospug_research/` on the host → `/root/rospug_research/` inside the container.
All scripts in `rospug_research/scripts/` are immediately available inside the container.

---

## 4. ROS Networking Architecture

### Why ROS commands can't run locally in the container

The robot's `ROS_HOSTNAME=localhost` means:
- roscore listens only on `127.0.0.1:11311` (not `192.168.123.1:11311`)
- All robot nodes (publishers, service servers) advertise `localhost:PORT` as their endpoint
- When the container queries rosmaster for a topic publisher, it gets `localhost:PORT`
- The container then connects to its own `localhost`, not the robot → **connection fails**

### What works vs. what does not from the container

| Operation | From Container | Via SSH on Robot |
|-----------|---------------|-----------------|
| `rostopic list` | ❌ (can't reach master) | ✅ |
| `rostopic pub` to robot | ❌ (same reason) | ✅ |
| `rostopic echo` (subscribe) | ❌ (publisher at localhost) | ✅ |
| `rosservice call` | ❌ (server at localhost) | ✅ |

### Solution used: SSH orchestration pattern

All ROS commands execute on the robot via SSH. The container handles prompts, timing, and logic only.

```bash
# Pattern used in all scripts
robot_cmd() {
    sshpass -p "$ROBOT_PASS" ssh -o StrictHostKeyChecking=no hiwonder@192.168.123.1 \
        "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh; $*"
}
```

`sshpass` is installed automatically by the teleop script on first run.

---

## 5. Teleop Test Module

### Run the teleop test

```bash
# Step 1 — Start robot bringup (laptop terminal)
ssh hiwonder@192.168.123.1 \
  "nohup zsh -l -c 'roslaunch pug_bringup base.launch' > /tmp/bringup.log 2>&1 &"

# Step 2 — Start real-robot container
HOST_ETH_IP=192.168.123.100 docker compose -f ~/rospug_research/docker-compose.real.yml run --rm rospug bash

# Step 3 — Inside container, run teleop test
bash /root/rospug_research/scripts/teleop_test.sh
# Enter SSH password when prompted: hiwonder
```

### What the teleop test does

The script is interactive — press **Enter** at each prompt to proceed.

| Step | Action | ROS command |
|------|--------|-------------|
| 1 | Check topics | `rostopic list` via SSH |
| 2 | Battery check | `rostopic echo /ros_robot_controller/battery` via SSH |
| 3 | Stand up | `rosservice call /pug_control/go_home '{}'` via SSH |
| 4 | Set gait params | `rostopic pub /pug_control/gait pug_control/Gait {...}` via SSH |
| 5 | Set pose params | `rostopic pub /pug_control/pose pug_control/Pose {...}` via SSH |
| 6 | Walk forward 2 s | `timeout 2 rostopic pub -r 10 /pug_control/velocity_move ...` via SSH |
| 7 | Walk backward 2 s | Same with `x: -0.08` |
| 8 | Sit down | `rosservice call /pug_control/run_action_group '{name: "sit"}'` via SSH |
| 9 | Stand back up | `rosservice call /pug_control/go_home '{}'` via SSH |

### Walking parameters (confirmed working)

| Parameter | Value |
|-----------|-------|
| Forward speed | `x: 0.08 m/s` |
| Backward speed | `x: -0.08 m/s` |
| Yaw rate | `0.0 rad/s` |
| Stand height | `-0.13 m` |
| Swing time | `0.18 s` |
| Overlap time | `0.2 s` |

### Manual one-liner keyboard commands (via SSH on robot)

SSH into robot first:
```bash
ssh hiwonder@192.168.123.1
source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh
```

Then use these commands:

```bash
# Stand up
rosservice call /pug_control/go_home '{}'

# Walk forward (Ctrl+C to stop, then send stop command)
rostopic pub -r 10 /pug_control/velocity_move pug_control/Velocity "{x: 0.08, y: 0.0, yaw_rate: 0.0, stop: false}"

# Turn left
rostopic pub -r 10 /pug_control/velocity_move pug_control/Velocity "{x: 0.0, y: 0.0, yaw_rate: 0.3, stop: false}"

# Stop
rostopic pub -1 /pug_control/velocity_move pug_control/Velocity "{x: 0.0, y: 0.0, yaw_rate: 0.0, stop: true}"
```

---

## 6. Observation Bridge Verification

### Run obs_bridge_test.py

```bash
# Copy to robot and run
scp ~/rospug_research/scripts/obs_bridge_test.py hiwonder@192.168.123.1:~/
ssh hiwonder@192.168.123.1 "source /opt/ros/melodic/setup.zsh && python3 ~/obs_bridge_test.py --seconds 5"
```

### Confirmed results (robot in standby)

```
[OK]  /joint_states received
[OK]  /imu received
[BAT] Battery: 10065 mV  (OK)
[OK]  All 12 joints present
/joint_states rate : 10.4 Hz  (note: low — standby mode, see §9)
/imu rate          : 51.8 Hz  ✓
[PASS] Observation vector looks correct — ready for deploy_policy.py
```

### 26D observation vector layout

| Index | Content | Source topic |
|-------|---------|-------------|
| `obs[0:12]` | Joint positions (rad) — reordered to RL policy order | `/joint_states` |
| `obs[12:24]` | Joint velocities (rad/s) | `/joint_states` |
| `obs[24]` | Body roll (rad) | `/imu` |
| `obs[25]` | Body pitch (rad) | `/imu` |

### Joint order mapping

The robot publishes joints in `[lf, lb, rf, rb]` order. The RL policy expects `[rf, lf, rb, lb]` order.
`obs_bridge_test.py` handles the reordering automatically.

```python
JOINT_ORDER = ('rf_joint','rf_thigh','rf_calf',
               'lf_joint','lf_thigh','lf_calf',
               'rb_joint','rb_thigh','rb_calf',
               'lb_joint','lb_thigh','lb_calf')
```

### IMU topic

Use `/imu` (complementary filter output with real orientation), **not** `/ros_robot_controller/imu_raw` (always outputs zero quaternion — hardware limitation).

---

## 7. Quick Reconnect Reference (per session)

Run these 3 commands at the start of every session to confirm the robot is reachable:

```bash
# 1. Confirm Ethernet IP is set
ip addr show enp2s0 | grep "inet "
# Expected: inet 192.168.123.100/24

# 2. Ping robot
ping -c 3 192.168.123.1
# Expected: 0% packet loss, ~0.2ms RTT

# 3. Check ROS topics are live
ssh hiwonder@192.168.123.1 \
  "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh && \
   rostopic list 2>/dev/null | grep -E 'joint_states|imu|pug_control'"
# password: hiwonder
```

If step 3 shows no topics, the bringup hasn't started — run the bringup command from §2.

If step 1 shows no `192.168.123.100` address (after reboot):
```bash
sudo ip addr add 192.168.123.100/24 dev enp2s0
```

---

## 8. Infrastructure Files

All files are in `/home/arc09/rospug_research/`.

| File | Purpose |
|------|---------|
| `docker-compose.real.yml` | Docker container config for real-robot connection (separate from simulation) |
| `scripts/teleop_test.sh` | Interactive teleop verification: stand → walk forward → walk backward → sit → stand |
| `scripts/obs_bridge_test.py` | Validates the 26D RL observation vector from real hardware topics |
| `find_robot_ip.sh` | Network discovery script (now superseded — robot IP is `192.168.123.1`) |
| `robot_bringup_check.sh` | SSH-based bringup diagnostic (requires passwordless SSH — use manual check instead) |

---

## 9. Step 8 Deployment Notes

### Known limitations to resolve before running deploy_policy.py

**1. `joint_states` publishes at 10.4 Hz in standby**

During the obs_bridge_test, the robot was in standby (motors not active). In standby, the hardware controller publishes at ~10 Hz. Once the full bringup is active and the robot is standing with motors engaged, the rate should increase. The RL policy requires 50 Hz.

**Action for Step 8:** After starting bringup and commanding the robot to stand, re-run `obs_bridge_test.py` to confirm the rate increases.

**2. Joint positions are all 0.0 in standby**

Real joint angles only appear when the robot's motor controller is active. Zero positions in standby are expected and not a bug.

**3. Robot Python version is 3.6 (Ubuntu 18.04)**

`stable-baselines3` requires Python 3.7+. For `deploy_policy.py`, options are:
- Install Python 3.8 on robot via deadsnakes PPA: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.8`
- Export policy to ONNX and use `onnxruntime` (Python 3.6 compatible)

**4. All policy deployment runs directly on the robot via SSH**

Because `ROS_HOSTNAME=localhost` blocks remote ROS, `deploy_policy.py` must run on the robot (not in the container). Use:
```bash
scp ~/rospug_research/scripts/deploy_policy.py hiwonder@192.168.123.1:~/
scp ~/rospug_research/checkpoints/policy_B_500k.zip hiwonder@192.168.123.1:~/
ssh -t hiwonder@192.168.123.1 "python3.8 ~/deploy_policy.py --policy ~/policy_B_500k.zip --max-seconds 30"
```

### Safety checklist before first policy run

- [ ] Support string attached overhead
- [ ] Clear floor space (1 m radius)
- [ ] Battery above 9000 mV
- [ ] `obs_bridge_test.py` PASS with active bringup (not just standby)
- [ ] First run: `--max-seconds 2` only
- [ ] Watchdog active: `|roll| > 0.7 rad` stops all joints
