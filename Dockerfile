FROM osrf/ros:melodic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. Supplemental ROS + system packages
#    ros:melodic-desktop-full already contains: rviz, gazebo9, ros-control,
#    gazebo_ros_pkgs, robot_state_publisher, joint_state_publisher, tf2, etc.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-catkin-tools \
    ros-melodic-navigation \
    ros-melodic-gmapping \
    ros-melodic-cv-bridge \
    ros-melodic-image-transport \
    ros-melodic-laser-filters \
    ros-melodic-teb-local-planner \
    ros-melodic-robot-localization \
    ros-melodic-joy \
    ros-melodic-teleop-twist-keyboard \
    ros-melodic-ros-controllers \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Python runtime deps (hardware-independent subset)
#    opencv-python is intentionally omitted — the Jetson needs a CUDA build.
#    On this PC image cv2 is provided by ros-melodic-cv-bridge's system OpenCV.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir pyserial numpy

# ---------------------------------------------------------------------------
# 3. rosdep initialisation (base image may already have run rosdep init)
# ---------------------------------------------------------------------------
RUN rosdep init 2>/dev/null || true
RUN rosdep update

# ---------------------------------------------------------------------------
# 4. Clone ROSPug (single branch, shallow clone to keep image lean)
# ---------------------------------------------------------------------------
WORKDIR /root/catkin_ws/src
RUN git clone --depth 1 -b Jetson_nano_ros1 \
        https://github.com/Hiwonder/ROSpug.git

# ---------------------------------------------------------------------------
# 5. Exclude packages that require Jetson/ARM-specific binaries at build time
#    pug_app         -- ships a compiled TensorRT .engine + ARM libmyplugins.so
#    pug_peripherals -- requires gscam / CSI camera hardware
# ---------------------------------------------------------------------------
RUN touch ROSpug/src/pug_app/CATKIN_IGNORE \
          ROSpug/src/pug_peripherals/CATKIN_IGNORE

# ---------------------------------------------------------------------------
# 6. Install rosdep keys declared by the remaining packages
# ---------------------------------------------------------------------------
WORKDIR /root/catkin_ws
RUN bash -c "source /opt/ros/melodic/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y"

# ---------------------------------------------------------------------------
# 7. Build the workspace
# ---------------------------------------------------------------------------
RUN bash -c "source /opt/ros/melodic/setup.bash && \
    catkin_make -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/catkin_build.log"

# ---------------------------------------------------------------------------
# 8. Persist ROS environment for interactive shells
# ---------------------------------------------------------------------------
RUN echo "source /opt/ros/melodic/setup.bash"          >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash"     >> /root/.bashrc && \
    echo "export ROS_MASTER_URI=http://localhost:11311" >> /root/.bashrc && \
    echo "export ROS_HOSTNAME=localhost"                >> /root/.bashrc

# ---------------------------------------------------------------------------
# 9. Python 3.8 + RL tooling
#    Installed alongside Python 2.7/3.6 (system defaults for ROS Melodic).
#    Does NOT replace python3 — catkin build remains on Python 2.7/3.6.
#
#    Key problem: source /opt/ros/melodic/setup.bash sets PYTHONPATH to
#    .../python2.7/dist-packages, so python3.8 would otherwise import the
#    wrong (Python-2.7) rospy and fail on 'rospkg' not found.
#    Fix: install python3-rospy (puts correct Python 3 rospy into
#    /opt/ros/melodic/lib/python3/dist-packages) and register that path
#    via a .pth file in Python 3.8 site-packages — only python3.8 is
#    affected; Python 2.7 ROS tools (roslaunch etc.) are unchanged.
# ---------------------------------------------------------------------------

# 9a. Python 3.8 interpreter
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-distutils && \
    rm -rf /var/lib/apt/lists/*

# 9b. pip for Python 3.8 + RL stack
#     rospkg and catkin-pkg are pure Python and available on PyPI — they
#     provide Python 3.8 with the ROS utility packages rospy depends on.
#     (python3-rospy is not in the Melodic frozen snapshot; rospy itself
#     is already present in /opt/ros/melodic/lib/python3/dist-packages/
#     from the base image's ros-melodic-desktop-full install.)
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o /tmp/get-pip.py && \
    python3.8 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# Install PyTorch with CUDA 11.8 first; SB3 auto-detects GPU via torch.cuda.is_available()
# On CPU-only hosts, CUDA kernels simply won't load — training still works on CPU.
RUN python3.8 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu118

RUN python3.8 -m pip install --no-cache-dir \
    "gymnasium>=0.27" \
    "stable-baselines3>=2.0" \
    "tensorboard>=2.12" \
    "protobuf>=3.20,<4.0" \
    rospkg catkin-pkg \
    netifaces

# Bake the Python 3 ROS path into .bashrc so every `docker exec` shell
# gets the correct PYTHONPATH even after setup.bash resets it.
# Must appear AFTER the catkin .bashrc block (step 8) so it appends last.
RUN echo "" >> /root/.bashrc && \
    echo "# Python 3.x: use python3/dist-packages rospy (not python2.7 version)" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/melodic/lib/python3/dist-packages:\${PYTHONPATH}" >> /root/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
