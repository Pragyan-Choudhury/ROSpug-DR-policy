#!/bin/bash
# Launch the Gazebo simulation of ROSPug inside the Docker container.
# Run from the rospug_research/ directory.
set -e

# Grant Docker access to the local X11 display server
xhost +local:docker >/dev/null 2>&1 || {
    echo "WARNING: xhost failed -- Gazebo GUI may not appear."
    echo "Try:  sudo apt install x11-xserver-utils  and re-run."
}

echo "[rospug] Launching Gazebo simulation (pug_description)..."
docker compose run --rm rospug \
    roslaunch pug_description gazebo.launch
