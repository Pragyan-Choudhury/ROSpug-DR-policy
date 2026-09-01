#!/usr/bin/env bash
# robot_bringup_check.sh — SSH into the ROSPug and verify/start its ROS stack.
#
# Run on the HOST (not inside Docker).
# Usage:  bash robot_bringup_check.sh <robot_ip>
# Example: bash robot_bringup_check.sh 192.168.149.1

set -euo pipefail

ROBOT_IP="${1:?Usage: $0 <robot_ip>}"
ROBOT_USER="hiwonder"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"

# ── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo "============================================================"
echo "  ROSPug Robot-side Bringup Check"
echo "  Target: $ROBOT_USER@$ROBOT_IP"
echo "============================================================"
echo ""

# ── 1. SSH reachability ────────────────────────────────────────────────────
echo "[1/6] SSH connectivity ..."
if ! ssh $SSH_OPTS "$ROBOT_USER@$ROBOT_IP" "echo connected" &>/dev/null; then
    fail "Cannot SSH to $ROBOT_USER@$ROBOT_IP"
    echo "     Try: ssh $ROBOT_USER@$ROBOT_IP  (default password: hiwonder)"
fi
ok "SSH to robot"

# From here, all checks run via SSH
run_on_robot() {
    ssh $SSH_OPTS "$ROBOT_USER@$ROBOT_IP" "$@"
}

# ── 2. Serial port ─────────────────────────────────────────────────────────
echo ""
echo "[2/6] Serial port /dev/rrc (STM32 bridge) ..."
if run_on_robot "test -e /dev/rrc"; then
    RRC_TARGET=$(run_on_robot "readlink -f /dev/rrc" 2>/dev/null || echo "?")
    ok "/dev/rrc → $RRC_TARGET"
else
    fail "/dev/rrc does not exist — STM32 driver not ready"
fi

# ── 3. Battery check ───────────────────────────────────────────────────────
echo ""
echo "[3/6] ROS processes ..."
ROSCORE_RUNNING=$(run_on_robot "pgrep -c roscore" 2>/dev/null || echo "0")
PUGNODE_RUNNING=$(run_on_robot "pgrep -fc pug_node" 2>/dev/null || echo "0")
RRC_RUNNING=$(run_on_robot "pgrep -fc ros_robot_controller_node" 2>/dev/null || echo "0")

if [[ "$ROSCORE_RUNNING" -gt 0 ]]; then
    ok "roscore is running"
else
    warn "roscore is NOT running"
fi
if [[ "$PUGNODE_RUNNING" -gt 0 ]]; then
    ok "pug_node.py is running"
else
    warn "pug_node.py is NOT running"
fi
if [[ "$RRC_RUNNING" -gt 0 ]]; then
    ok "ros_robot_controller_node.py is running"
else
    warn "ros_robot_controller_node.py is NOT running"
fi

# ── 4. Start bringup if needed ────────────────────────────────────────────
echo ""
if [[ "$ROSCORE_RUNNING" -eq 0 || "$PUGNODE_RUNNING" -eq 0 ]]; then
    echo "[4/6] Bringup stack is not running. Starting base.launch ..."
    echo "      (this will take ~15 seconds)"
    # Start in background on robot; use nohup so it persists after SSH exits
    run_on_robot "bash -lc 'source ~/catkin_ws/devel/setup.bash && \
        nohup roslaunch pug_bringup base.launch > /tmp/rospug_bringup.log 2>&1 &' "
    sleep 15
    ok "base.launch started — check /tmp/rospug_bringup.log on robot if issues arise"
else
    ok "[4/6] Bringup stack already running — no action needed"
fi

# ── 5. Topic availability check ───────────────────────────────────────────
echo ""
echo "[5/6] Checking required topics ..."
REQUIRED=(
    "/joint_states"
    "/imu"
    "/ros_robot_controller/imu_raw"
    "/ros_robot_controller/battery"
    "/pug_control/velocity_move"
)
for topic in "${REQUIRED[@]}"; do
    if run_on_robot "source ~/catkin_ws/devel/setup.bash && \
        timeout 3 rostopic echo -n 1 $topic &>/dev/null" 2>/dev/null; then
        ok "$topic"
    else
        warn "$topic — not receiving data"
    fi
done

# ── 6. Available action groups ────────────────────────────────────────────
echo ""
echo "[6/6] Available action groups (for sit/stand commands) ..."
ACTION_DIR=$(run_on_robot "find ~/catkin_ws -name '*.d6a' -o -name '*.yaml' 2>/dev/null | \
    xargs grep -l 'action' 2>/dev/null | head -3" 2>/dev/null || echo "")
if [[ -n "$ACTION_DIR" ]]; then
    echo "  Action files found: $ACTION_DIR"
else
    warn "Could not find action group files — 'sit' action group may not exist"
    echo "  Fallback: use Pose topic with height=0.0 to approximate sit"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
ok "Robot-side check complete."
echo ""
echo "  Next step: start the real-robot container from your laptop:"
echo ""
echo "    export ROBOT_IP=$ROBOT_IP"
echo "    export HOST_ETH_IP=\$(ip addr show enp2s0 | grep 'inet ' | head -1 | awk '{print \$2}' | cut -d/ -f1)"
echo "    docker compose -f docker-compose.real.yml run --rm rospug bash"
echo ""
echo "  Then inside the container:"
echo "    bash /root/rospug_research/scripts/teleop_test.sh"
echo "============================================================"
