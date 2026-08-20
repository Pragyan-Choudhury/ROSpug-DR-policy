#!/usr/bin/env python3.8
"""
rospug_env.py — gymnasium.Env wrapper for ROSPug in Gazebo (ROS 1 Melodic)

Action space:  Box(-0.15, 0.15, shape=(12,), float32)
               Residual corrections added to a reference trot gait.
               The gait drives thigh joints sinusoidally in diagonal trot pairs
               (RF+LB in phase, LF+RB offset by half a cycle). PPO learns small
               corrections on all 12 joints on top of this baseline. Joint order:
                 [0]  rf_joint   [1]  rf_thigh  [2]  rf_calf
                 [3]  lf_joint   [4]  lf_thigh  [5]  lf_calf
                 [6]  rb_joint   [7]  rb_thigh  [8]  rb_calf
                 [9]  lb_joint   [10] lb_thigh  [11] lb_calf

Observation:   Box(-inf, inf, shape=(26,), float32)
                 [0:12]  joint positions  (rad)
                 [12:24] joint velocities (rad/s)
                 [24]    body roll        (rad)
                 [25]    body pitch       (rad)
               Body orientation comes from /gazebo/model_states quaternion
               (no IMU Gazebo plugin is present in the ROSPug URDF).

Reward:        vx * 3.0  +  0.5  -  10.0 * fallen
               Energy penalty removed: PPO's Gaussian exploration (std≈1) causes
               high joint velocities, making the energy term dominate and creating
               a perverse incentive to fall quickly (shorter episode = less energy
               penalty). Alive bonus 0.5/step strongly incentivises survival first,
               forward motion second.

Termination:   |roll| > 0.7 rad  OR  |pitch| > 0.7 rad  → terminated = True
Truncation:    step_count >= MAX_STEPS (500)              → truncated  = True

Prerequisites (run inside Docker container):
  Terminal 1:  roslaunch pug_description gazebo.launch   # press Play in GUI
  Terminal 2:  python3.8 /root/rospug_research/scripts/test_env_random.py
"""

import time
import threading
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import math

import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float):
    """Convert quaternion to (roll, pitch, yaw) in radians — no tf dependency."""
    # roll (x-axis)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    # yaw (z-axis)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed action/observation joint ordering — must stay consistent.
JOINT_ORDER: Tuple[str, ...] = (
    'rf_joint', 'rf_thigh', 'rf_calf',
    'lf_joint', 'lf_thigh', 'lf_calf',
    'rb_joint', 'rb_thigh', 'rb_calf',
    'lb_joint', 'lb_thigh', 'lb_calf',
)

#: ROS position-controller command topic for each joint.
#: Derived from pug_description/config/gazebo_control.yaml controller names.
JOINT_TOPICS: Dict[str, str] = {
    'rf_joint':  '/pug/rf_joint_position_controller/command',
    'rf_thigh':  '/pug/rf_thigh_position_controller/command',
    'rf_calf':   '/pug/rf_calf_position_controller/command',
    'lf_joint':  '/pug/lf_joint_position_controller/command',
    'lf_thigh':  '/pug/lf_thigh_position_controller/command',
    'lf_calf':   '/pug/lf_calf_position_controller/command',
    'rb_joint':  '/pug/rb_joint_position_controller/command',
    'rb_thigh':  '/pug/rb_thigh_position_controller/command',
    'rb_calf':   '/pug/rb_calf_position_controller/command',
    'lb_joint':  '/pug/lb_joint_position_controller/command',
    'lb_thigh':  '/pug/lb_thigh_position_controller/command',
    'lb_calf':   '/pug/lb_calf_position_controller/command',
}

#: Standing pose: all joints at 0.0 rad (verified from sim_gait_controller_v3.py).
STAND_POSE = np.zeros(12, dtype=np.float32)

# ---------------------------------------------------------------------------
# Reference trot gait  (kinematics from sim_gait_controller_v3.py)
# ---------------------------------------------------------------------------
# Diagonal trot pairs: RF+LB swing together (phase 0.0), LF+RB offset by 0.5 cycle.
# Right-leg thigh axis [0,-1,0] → positive angle = foot forward  → THIGH_SIGN = +1
# Left-leg  thigh axis [0,+1,0] → positive angle = foot backward → THIGH_SIGN = -1
# THIGH_SIGN normalises so the same positive amplitude drives "foot forward" on all legs.
GAIT_FREQ    = 1.5    # Hz — trot cycle frequency (matches sim_gait_controller_v3.py)
THIGH_AMP    = 0.20   # rad — sagittal thigh amplitude (gait controller uses 0.30 at vx=1.0)

_THIGH_SIGN: Dict[str, int] = {'rf': +1, 'lf': -1, 'rb': +1, 'lb': -1}
_PHASE_BIAS: Dict[str, float] = {'rf': 0.0, 'lf': 0.5, 'rb': 0.5, 'lb': 0.0}

# Pre-parsed (leg, joint_type) for every slot in JOINT_ORDER — avoids per-step string ops.
# Populated after JOINT_ORDER is defined (see line below).
_JOINT_GAIT: Tuple  # filled after JOINT_ORDER

GAIT_RESIDUAL = 0.15   # rad — maximum PPO residual correction on top of gait reference
ACTION_LIMIT  = GAIT_RESIDUAL  # alias so action_space init needs no change

CTRL_RATE     = 50.0   # Hz
DT            = 1.0 / CTRL_RATE   # 0.02 s per control step (wall clock)
MAX_STEPS     = 500    # 10 s per episode at 50 Hz
FALL_THRESH   = 0.7    # rad ≈ 40° — termination threshold on |roll| or |pitch|
RESET_SETTLE  = 2.0    # wall-clock seconds to let Gazebo physics settle after reset
                       # (increased from 1.0: rapid-fall training corrupts physics state)
STAND_SETTLE  = 0.5    # wall-clock seconds for stand-pose stabilisation
DATA_TIMEOUT  = 10.0   # seconds to wait for first ROS topic messages

# Fill in _JOINT_GAIT now that JOINT_ORDER is defined.
_JOINT_GAIT = tuple((jn.split('_')[0], jn.split('_')[1]) for jn in JOINT_ORDER)


# ---------------------------------------------------------------------------
# RosPugEnv
# ---------------------------------------------------------------------------

class RosPugEnv(gym.Env):
    """
    gymnasium.Env wrapping the ROSPug 12-DOF quadruped in Gazebo (ROS 1 Melodic).

    Usage::

        env = RosPugEnv()
        obs, info = env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
    """

    metadata = {'render_modes': []}

    def __init__(self, node_name: str = 'rospug_env') -> None:
        super().__init__()

        # --- Gymnasium spaces ---
        self.action_space = spaces.Box(
            low=-ACTION_LIMIT, high=ACTION_LIMIT, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        # --- ROS node ---
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=False, disable_signals=True)

        # --- Joint publishers ---
        self._pubs: Dict[str, rospy.Publisher] = {
            name: rospy.Publisher(topic, Float64, queue_size=1)
            for name, topic in JOINT_TOPICS.items()
        }

        # --- Thread-safe subscriber state ---
        self._lock = threading.Lock()
        self._js_data: Optional[JointState] = None
        self._ms_data: Optional[ModelStates] = None
        self._model_idx: Optional[int] = None  # index of 'pug' in ModelStates

        rospy.Subscriber('/pug/joint_states', JointState,
                         self._js_callback, queue_size=1, buff_size=2**20)
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self._ms_callback, queue_size=1, buff_size=2**20)

        # --- Gazebo services ---
        rospy.wait_for_service('/gazebo/reset_world', timeout=15.0)
        self._svc_reset = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        rospy.wait_for_service('/gazebo/unpause_physics', timeout=15.0)
        self._svc_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # --- Episode state ---
        self._step_count = 0
        self._gait_time  = 0.0   # seconds since episode start; drives trot phase

        rospy.loginfo('RosPugEnv: initialised. Call reset() to begin.')

    # ------------------------------------------------------------------
    # Subscriber callbacks (run in ROS spin thread)
    # ------------------------------------------------------------------

    def _js_callback(self, msg: JointState) -> None:
        with self._lock:
            self._js_data = msg

    def _ms_callback(self, msg: ModelStates) -> None:
        with self._lock:
            self._ms_data = msg
            # Discover robot model index on first message.
            # Gazebo spawns the robot as '-model pug' (see gazebo.launch).
            if self._model_idx is None:
                for i, name in enumerate(msg.name):
                    if 'pug' in name.lower() and 'ground' not in name.lower():
                        self._model_idx = i
                        rospy.loginfo(
                            "RosPugEnv: found robot model '%s' at ModelStates index %d",
                            name, i)
                        break

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # 1. Reset Gazebo world state (does NOT re-pause the simulation)
        self._svc_reset()

        # 2. Ensure simulation is running (unpause is idempotent if already running)
        try:
            self._svc_unpause()
        except rospy.ServiceException:
            pass  # already unpaused — not an error

        # 3. Physics settle (wall-clock avoids use_sim_time hang if sim paused)
        time.sleep(RESET_SETTLE)

        # 4. Command stand pose so robot is upright before episode begins
        self._publish_joints(STAND_POSE)
        time.sleep(STAND_SETTLE)

        # 5. Wait for valid topic data (first-run guard)
        self._wait_for_data()

        self._step_count = 0
        self._gait_time  = 0.0   # reset gait phase at episode start
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Reference trot gait + small PPO residual correction
        #    The gait provides coordinated leg motion; PPO refines it.
        action = np.clip(action, -GAIT_RESIDUAL, GAIT_RESIDUAL).astype(np.float32)
        targets = self._gait_targets() + action
        self._publish_joints(targets)
        self._gait_time += DT   # advance gait phase by one control step

        # 2. Wait one control cycle (wall clock — independent of use_sim_time)
        time.sleep(DT)

        # 3. Build observation
        obs = self._get_obs()

        # 4. Reward and termination
        reward, terminated = self._compute_reward(obs)
        self._step_count += 1
        truncated = (self._step_count >= MAX_STEPS)

        info: Dict[str, Any] = {
            'step':  self._step_count,
            'vx':    float(self._get_vx()),
            'roll':  float(obs[24]),
            'pitch': float(obs[25]),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        rospy.signal_shutdown('RosPugEnv.close() called')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_joints(self, targets: np.ndarray) -> None:
        """Publish 12 Float64 joint position commands."""
        for i, name in enumerate(JOINT_ORDER):
            self._pubs[name].publish(Float64(float(targets[i])))

    def _gait_targets(self) -> np.ndarray:
        """
        Reference trot joint targets for the current gait time.

        Only thigh joints are driven sinusoidally — hip abduction (joint) and
        calf joints are left at 0 so PPO learns to control them via residuals.

        Sign convention from sim_gait_controller_v3.py URDF analysis:
          Right legs axis [0,-1,0]: positive angle = foot forward → THIGH_SIGN = +1
          Left  legs axis [0,+1,0]: positive angle = foot backward → THIGH_SIGN = -1
        """
        phase = 2.0 * math.pi * GAIT_FREQ * self._gait_time
        targets = np.zeros(12, dtype=np.float32)
        for i, (leg, jtype) in enumerate(_JOINT_GAIT):
            if jtype == 'thigh':
                leg_phase = phase + 2.0 * math.pi * _PHASE_BIAS[leg]
                targets[i] = float(_THIGH_SIGN[leg]) * THIGH_AMP * math.sin(leg_phase)
            # 'joint' (hip abduction) and 'calf': 0 — PPO residuals control these
        return targets

    def _get_obs(self) -> np.ndarray:
        """Return 26D observation vector from latest ROS topic data."""
        with self._lock:
            js = self._js_data
            ms = self._ms_data
            idx = self._model_idx

        # Joint positions and velocities — reorder to match JOINT_ORDER
        pos = np.zeros(12, dtype=np.float32)
        vel = np.zeros(12, dtype=np.float32)
        if js is not None:
            name_to_i = {n: i for i, n in enumerate(js.name)}
            for k, jname in enumerate(JOINT_ORDER):
                j = name_to_i.get(jname)
                if j is not None:
                    pos[k] = float(js.position[j])
                    vel[k] = float(js.velocity[j])

        # Body orientation from Gazebo model states
        roll, pitch = 0.0, 0.0
        if ms is not None and idx is not None:
            q = ms.pose[idx].orientation
            roll, pitch, _ = _quat_to_rpy(q.x, q.y, q.z, q.w)

        return np.concatenate([pos, vel,
                                np.array([roll, pitch], dtype=np.float32)]).astype(np.float32)

    def _get_vx(self) -> float:
        """Forward (x-axis) body velocity in the world frame from Gazebo."""
        with self._lock:
            ms = self._ms_data
            idx = self._model_idx
        if ms is None or idx is None:
            return 0.0
        return float(ms.twist[idx].linear.x)

    def _compute_reward(self, obs: np.ndarray) -> Tuple[float, bool]:
        """
        Returns (reward, terminated).

        reward = 3*vx + 0.5 - 10.0 * fallen
          vx   : forward body velocity (m/s) — primary locomotion signal
          0.5  : alive bonus per step — survival is always better than falling
          fallen: True if |roll| > FALL_THRESH or |pitch| > FALL_THRESH

        Energy penalty deliberately removed: PPO's Gaussian exploration (std≈1.0)
        generates high joint velocities, causing the energy penalty to dominate and
        creating a perverse "fall fast" optimum (short episode → less accumulated
        energy penalty → higher reward). Re-add energy penalty in Step 5 once a
        walking gait has been established.

        Reward per episode:
          stand still (500 steps, no fall):  500 × 0.5           = +250
          walk 0.2 m/s (500 steps, no fall): 500 × (0.6 + 0.5)  = +550
          fall at step 21:                   21 × 0.5 - 10.0     =  -9.5
        """
        roll  = float(obs[24])
        pitch = float(obs[25])
        fallen = (abs(roll) > FALL_THRESH) or (abs(pitch) > FALL_THRESH)

        vx = self._get_vx()
        reward = vx * 3.0 + 0.5
        if fallen:
            reward -= 10.0

        return float(reward), bool(fallen)

    def _wait_for_data(self, timeout: float = DATA_TIMEOUT) -> None:
        """Block until both /pug/joint_states and /gazebo/model_states have arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._js_data is not None and self._ms_data is not None:
                    return
            time.sleep(0.05)
        raise TimeoutError(
            'RosPugEnv: timed out waiting for /pug/joint_states and '
            '/gazebo/model_states. Is Gazebo running and Play pressed?'
        )
