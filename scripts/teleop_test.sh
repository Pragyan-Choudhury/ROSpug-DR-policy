#!/usr/bin/env bash
# teleop_test.sh — Step-by-step teleoperation verification for real ROSPug
#
# Run INSIDE the Docker container started with docker-compose.real.yml.
#
# How it works:
#   ALL ROS commands run on the robot via SSH (robot's roscore only listens
#   on localhost:11311 and is not reachable from outside). The container
#   handles prompts, timing, and orchestration only.
#
# Prerequisites:
#   1. Robot powered on, bringup running:
#        ssh hiwonder@192.168.123.1 "nohup zsh -l -c 'roslaunch pug_bringup base.launch' > /tmp/bringup.log 2>&1 &"
#   2. Container started:
#        HOST_ETH_IP=192.168.123.100 docker compose -f docker-compose.real.yml run --rm rospug bash
#   3. Run: bash /root/rospug_research/scripts/teleop_test.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-192.168.123.1}"
ROBOT_USER="hiwonder"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5"

# ── Colour helpers ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!!]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
STEP=0
step() { STEP=$((STEP+1)); echo ""; echo -e "${YELLOW}── Step $STEP: $* ──${NC}"; }

# ── SSH helper — ALL ROS commands run here ────────────────────────────────
robot_cmd() {
    sshpass -p "$ROBOT_PASS" ssh $SSH_OPTS "$ROBOT_USER@$ROBOT_IP" \
        "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh; $*"
}

# ── Nominal walking parameters ────────────────────────────────────────────
OVERLAP_TIME=0.2
SWING_TIME=0.18
CLEARANCE_TIME=0.02
Z_CLEARANCE=0.04
STAND_HEIGHT=-0.13
X_SHIFT=0.005
FWD_VX=0.08
BWD_VX=-0.08
WALK_DURATION=2

echo "============================================================"
echo "  ROSPug Hardware Teleoperation Verification"
echo "  ROBOT_IP : $ROBOT_IP"
echo "  (All ROS commands run on robot via SSH)"
echo "============================================================"

# ── Install sshpass if not present ───────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    echo "Installing sshpass..."
    apt-get update -q && apt-get install -y -q sshpass || \
        fail "Cannot install sshpass — run: apt-get update && apt-get install sshpass"
fi

# ── Prompt for robot SSH password once ───────────────────────────────────
read -rsp "  Robot SSH password (default: hiwonder): " ROBOT_PASS
ROBOT_PASS="${ROBOT_PASS:-hiwonder}"
echo ""

# ── 0. Connectivity check (via SSH) ──────────────────────────────────────
step "ROS master + topic connectivity"
if ! robot_cmd "rostopic list > /dev/null 2>&1"; then
    fail "Cannot reach robot roscore via SSH — is base.launch running on the robot?"
fi
ok "ROS master reachable"

for topic in /joint_states /pug_control/velocity_move /ros_robot_controller/battery; do
    robot_cmd "rostopic list 2>/dev/null | grep -q '^${topic}$'" \
        && ok "$topic" || warn "$topic not found — proceeding anyway"
done

step "Battery check"
BATTERY=$(robot_cmd "rostopic echo -n 1 /ros_robot_controller/battery 2>/dev/null | grep 'data:' | awk '{print \$2}'" 2>/dev/null || echo "N/A")
if [[ -z "$BATTERY" || "$BATTERY" == "N/A" ]]; then
    warn "Could not read battery — proceeding"
else
    ok "Battery: ${BATTERY} mV"
    [[ "$BATTERY" -lt 7000 ]] 2>/dev/null && fail "Battery critically low (${BATTERY} mV)"
fi

# ── 1. Stand up ───────────────────────────────────────────────────────────
step "Stand up via go_home service"
echo "  ⚠  Place robot on a flat surface before pressing Enter."
read -rp "  Press Enter to stand up ..."
robot_cmd "rosservice call /pug_control/go_home '{}'" && ok "go_home OK" || \
    warn "go_home not available — robot may already be standing"
sleep 2

# ── 2. Set gait/pose parameters ───────────────────────────────────────────
step "Set nominal Gait + Pose parameters"
robot_cmd "rostopic pub -1 /pug_control/gait pug_control/Gait \
    '{overlap_time: $OVERLAP_TIME, swing_time: $SWING_TIME, clearance_time: $CLEARANCE_TIME, z_clearance: $Z_CLEARANCE}'"
robot_cmd "rostopic pub -1 /pug_control/pose pug_control/Pose \
    '{roll: 0.0, pitch: 0.0, yaw: 0.0, height: $STAND_HEIGHT, x_shift: $X_SHIFT, stance_x: 0.0, stance_y: 0.0, run_time: 0.5}'"
sleep 1
ok "Gait and pose parameters set"

# ── 3. Walk forward ───────────────────────────────────────────────────────
step "Walk FORWARD for ${WALK_DURATION} s  (vx=${FWD_VX} m/s)"
echo "  ⚠  Ensure ~0.5 m clear space in front of robot."
read -rp "  Press Enter to start ..."
robot_cmd "rostopic pub -r 10 /pug_control/velocity_move pug_control/Velocity \
    '{x: ${FWD_VX}, y: 0.0, yaw_rate: 0.0, stop: false}' &
_WALK_PID=\$!; sleep ${WALK_DURATION}
kill \$_WALK_PID 2>/dev/null || true; wait \$_WALK_PID 2>/dev/null || true
rostopic pub -1 /pug_control/velocity_move pug_control/Velocity '{x: 0.0, y: 0.0, yaw_rate: 0.0, stop: true}'"
ok "Forward walk done"
sleep 1

# ── 4. Walk backward ─────────────────────────────────────────────────────
step "Walk BACKWARD for ${WALK_DURATION} s  (vx=${BWD_VX} m/s)"
read -rp "  Press Enter to start ..."
robot_cmd "rostopic pub -r 10 /pug_control/velocity_move pug_control/Velocity \
    '{x: ${BWD_VX}, y: 0.0, yaw_rate: 0.0, stop: false}' &
_WALK_PID=\$!; sleep ${WALK_DURATION}
kill \$_WALK_PID 2>/dev/null || true; wait \$_WALK_PID 2>/dev/null || true
rostopic pub -1 /pug_control/velocity_move pug_control/Velocity '{x: 0.0, y: 0.0, yaw_rate: 0.0, stop: true}'"
ok "Backward walk done"
sleep 1

# ── 5. Sit down ───────────────────────────────────────────────────────────
step "Sit down"
read -rp "  Press Enter to sit ..."
robot_cmd "rosservice call /pug_control/run_action_group '{name: \"sit\"}' 2>/dev/null | grep -q 'True'" \
    && ok "Sit action executed" || {
    warn "No sit action group — lowering via Pose"
    robot_cmd "rostopic pub -1 /pug_control/pose pug_control/Pose \
        '{roll: 0.0, pitch: 0.0, yaw: 0.0, height: 0.0, x_shift: 0.0, stance_x: 0.0, stance_y: 0.0, run_time: 1.0}'"
}
sleep 2

# ── 6. Stand back up ─────────────────────────────────────────────────────
step "Stand back up"
read -rp "  Press Enter to stand up ..."
robot_cmd "rosservice call /pug_control/go_home '{}' 2>/dev/null" && ok "Standing" || \
    robot_cmd "rostopic pub -1 /pug_control/pose pug_control/Pose \
        '{roll: 0.0, pitch: 0.0, yaw: 0.0, height: $STAND_HEIGHT, x_shift: $X_SHIFT, stance_x: 0.0, stance_y: 0.0, run_time: 0.5}'"
sleep 2
ok "Stand up complete"

# ── 7. Sensor readback ────────────────────────────────────────────────────
step "Live sensor verification"
echo "  Joint states (positions):"
robot_cmd "timeout 5 rostopic echo -n 1 /joint_states 2>/dev/null | grep -A 15 'position:' | head -6" \
    || warn "Could not read /joint_states"
echo ""
echo "  IMU orientation:"
robot_cmd "timeout 5 rostopic echo -n 1 /imu 2>/dev/null | grep -A 4 'orientation:'" \
    || warn "Could not read /imu"

echo ""
echo "============================================================"
ok "Teleoperation verification complete!"
echo "============================================================"

