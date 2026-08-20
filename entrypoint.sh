#!/bin/bash
# Container entrypoint — sources the full ROS + catkin environment
# before executing any command passed to docker run / docker-compose run.
set -e

source /opt/ros/melodic/setup.bash
source /root/catkin_ws/devel/setup.bash

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost

# Prepend the Python 3 ROS packages so python3.x finds rospy from
# python3/dist-packages BEFORE the python2.7/dist-packages that setup.bash
# adds to PYTHONPATH.  Python 2.7 ROS tools (roslaunch, catkin, etc.) are
# unaffected — they use /usr/bin/python whose sys.path is resolved separately.
export PYTHONPATH=/opt/ros/melodic/lib/python3/dist-packages:${PYTHONPATH}

exec "$@"
