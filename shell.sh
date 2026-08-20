#!/bin/bash
# Open an interactive ROS Melodic development shell inside the container.
# The catkin workspace is pre-sourced; use rosrun / roslaunch / catkin_make freely.
set -e

xhost +local:docker >/dev/null 2>&1 || true

echo "[rospug] Opening dev shell -- ROS Melodic + catkin_ws are sourced."
echo "         Workspace:        /root/catkin_ws"
echo "         ROSPug packages:  /root/catkin_ws/src/ROSpug/src/"
echo ""
docker compose run --rm rospug bash
