#!/usr/bin/env python3
"""
deploy_policy.py — Deploy a pre-trained RL walking policy on the real ROSPug.

Run ON THE ROBOT via SSH (not in Docker container — ROS_HOSTNAME=localhost blocks
remote ROS subscriptions).  Recommended invocation from a laptop terminal:

  ssh -t hiwonder@192.168.123.1 \\
    "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh && \\
     PYTHONPATH=/opt/ros/melodic/lib/python3/dist-packages:\\$PYTHONPATH \\
     python3 ~/deploy_policy.py --policy ~/policy_A.onnx --max-seconds 10"

Dependencies (one-time install on robot):
  pip3 install onnxruntime numpy     # ~25 MB total; onnxruntime has aarch64 wheels

Controls:
  Ctrl+C   → safe stop (send stand pose, then exit)
  Timeout  → same safe stop as Ctrl+C

Policies to deploy (in order):
  policy_A_500k.onnx  — nominal physics, no DR
  policy_B_500k.onnx  — fixed-range domain randomisation
  policy_C_500k.onnx  — SW-ADR adaptive domain randomisation (expected best)
"""

import sys
import os
import math
import time
import signal
import argparse
import threading

import numpy as np

import rospy
from sensor_msgs.msg import JointState, Imu

try:
    from ros_robot_controller.msg import SetJointAngle
except ImportError:
    sys.stderr.write(
        "[FATAL] Cannot import SetJointAngle.\n"
        "  Make sure you sourced ~/pug/devel/setup.zsh and set PYTHONPATH to include\n"
        "  /opt/ros/melodic/lib/python3/dist-packages\n"
    )
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    sys.stderr.write(
        "[FATAL] onnxruntime not found.  Install with:\n"
        "  pip3 install onnxruntime\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants mirrored exactly from rospug_env.py (DO NOT edit independently)
# ---------------------------------------------------------------------------

JOINT_ORDER = (
    'rf_joint', 'rf_thigh', 'rf_calf',
    'lf_joint', 'lf_thigh', 'lf_calf',
    'rb_joint', 'rb_thigh', 'rb_calf',
    'lb_joint', 'lb_thigh', 'lb_calf',
)

GAIT_FREQ     = 1.5    # Hz
THIGH_AMP     = 0.20   # rad
GAIT_RESIDUAL = 0.15   # rad — maximum residual action

# THIGH_SIGN = CALF_SIGN per URDF: right-leg positive = foot forward, left-leg positive = foot backward.
_THIGH_SIGN = {'rf': +1, 'lf': -1, 'rb': +1, 'lb': -1}
# Matches sim_gait_controller_v3: CALF_SIGN = THIGH_SIGN (half-wave, swing phase only).
_CALF_SIGN  = {'rf': +1, 'lf': -1, 'rb': +1, 'lb': -1}
# +0.5 shift from sim: hardware thigh direction is reversed, so phase 0 = hardware stance.
_PHASE_BIAS = {'rf': 0.5, 'lf': 0.0, 'rb': 0.0, 'lb': 0.5}

_JOINT_GAIT = tuple((jn.split('_')[0], jn.split('_')[1]) for jn in JOINT_ORDER)

FALL_THRESH = 0.7   # rad — |roll| or |pitch| above this → immediate safe stop

# Maximum joint velocity for velocity clamping (rad/s).
# At 10 Hz → 0.4 rad/step; at 50 Hz → 0.08 rad/step.
# Set conservatively above the max gait+residual rate of change.
MAX_JOINT_VEL = 4.0  # rad/s

# ---------------------------------------------------------------------------
# Sim-angle → servo-unit conversion
#
# SetJointAngle.joint_angle expects SERVO UNITS (0–1000), NOT radians.
# The SDK's set_joint_angle, set_leg_ik, etc. all operate in servo units.
#
# Derivation: inverse of pug_node.py get_robot_leg_ik_sim, which converts
# hardware servo readings back to URDF/sim angles for Gazebo.
#   sim_angle = (servo_raw - _SIM_OFFSET[i]) / 1000.0 * π * _SIM_SIGN[i]
#   ↔  servo_raw = sim_angle * _SIM_SIGN[i] * (1000/π) + _SIM_OFFSET[i]
#
# Verified: applying these offsets to the standing servo values returned by
#   rosservice call /ros_robot_controller/robot/set_leg_ik  (z=-0.13 m)
# reproduces the confirmed standing angles (sim: ±0.200 thighs, ±0.239 hips).
# ---------------------------------------------------------------------------

_SIM_SCALE = 1000.0 / math.pi   # servo units per radian ≈ 318.31

# Sign and offset per JOINT_ORDER slot (derived from get_robot_leg_ik_sim):
#   JOINT_ORDER = (rf_joint, rf_thigh, rf_calf,
#                  lf_joint, lf_thigh, lf_calf,
#                  rb_joint, rb_thigh, rb_calf,
#                  lb_joint, lb_thigh, lb_calf)
_SIM_SIGN   = np.array([  1,   1,  -1,   1,   1,   1,   1,   1,   1,   1,   1,  -1], dtype=np.float32)
_SIM_OFFSET = np.array([600, 250, 500, 400, 750, 500, 600, 250, 500, 400, 750, 500], dtype=np.float32)

# Natural standing pose in sim-angle space (sim angle that maps to the
# hardware's go_home Cartesian stance: x=0, y=0, z=-0.13 m per leg).
# At these angles the servo commands reproduce the confirmed standing servos:
#   calves ≈500, rf/rb_thigh≈314, lf/lb_thigh≈686, rf/rb_hip≈524, lf/lb_hip≈476
_STAND_OFFSET = np.array([
    -0.239,  0.200,  0.0,   # rf_joint, rf_thigh, rf_calf
     0.239, -0.200,  0.0,   # lf_joint, lf_thigh, lf_calf
    -0.239,  0.200,  0.0,   # rb_joint, rb_thigh, rb_calf
     0.239, -0.200,  0.0,   # lb_joint, lb_thigh, lb_calf
], dtype=np.float32)

# Reorder from JOINT_ORDER to SDK 3×4 flat (calf/thigh/hip rows × rf/lf/rb/lb cols):
#   SDK[0..3] calf:  policy[2,5,8,11]
#   SDK[4..7] thigh: policy[1,4,7,10]
#   SDK[8..11] hip:  policy[0,3,6,9]
_POLICY_TO_SDK = np.array([2, 5, 8, 11, 1, 4, 7, 10, 0, 3, 6, 9], dtype=np.intp)


# ---------------------------------------------------------------------------
# Quaternion → roll/pitch  (same formula as rospug_env.py — no tf dependency)
# ---------------------------------------------------------------------------

def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


# ---------------------------------------------------------------------------
# Gait reference — mirrors _gait_targets() in rospug_env.py,
# but uses wall-clock time instead of step-count accumulation so timing is
# correct regardless of the actual hardware control rate.
# ---------------------------------------------------------------------------

def _gait_targets(t: float) -> np.ndarray:
    """Return 12-D thigh-only gait reference."""
    phase   = 2.0 * math.pi * GAIT_FREQ * t
    targets = np.zeros(12, dtype=np.float32)
    for i, (leg, jtype) in enumerate(_JOINT_GAIT):
        if jtype == 'thigh':
            leg_phase  = phase + 2.0 * math.pi * _PHASE_BIAS[leg]
            targets[i] = float(_THIGH_SIGN[leg]) * THIGH_AMP * math.sin(leg_phase)
    return targets


def _calf_lift(t: float, calf_amp: float) -> np.ndarray:
    """Half-wave calf lift: non-zero only during swing, matching sim_gait_controller_v3."""
    if calf_amp <= 0.0:
        return np.zeros(12, dtype=np.float32)
    phase   = 2.0 * math.pi * GAIT_FREQ * t
    targets = np.zeros(12, dtype=np.float32)
    for i, (leg, jtype) in enumerate(_JOINT_GAIT):
        if jtype == 'calf':
            # +π/2 lead aligns calf active window with actual swing phase (not thigh phase)
            leg_phase  = phase + 2.0 * math.pi * _PHASE_BIAS[leg] + math.pi / 2
            targets[i] = float(_CALF_SIGN[leg]) * calf_amp * max(0.0, math.sin(leg_phase))
    return targets


# ---------------------------------------------------------------------------
# Thread-safe observation collector
# ---------------------------------------------------------------------------

class _ObsCollector:
    """Subscribes to /joint_states and /imu; builds the 26-D obs vector on demand."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._js        = None   # latest JointState
        self._imu       = None   # latest Imu
        self._js_count  = 0
        self._imu_count = 0

        rospy.Subscriber('/joint_states', JointState, self._js_cb,  queue_size=5)
        rospy.Subscriber('/imu',          Imu,        self._imu_cb, queue_size=5)

    def _js_cb(self, msg):
        with self._lock:
            self._js = msg
            self._js_count += 1

    def _imu_cb(self, msg):
        with self._lock:
            self._imu = msg
            self._imu_count += 1

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """Block until both topics have been received at least once."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._js is not None and self._imu is not None:
                    return True
            time.sleep(0.1)
        return False

    def build_obs(
        self,
        commanded_pos: 'np.ndarray | None' = None,
        commanded_vel: 'np.ndarray | None' = None,
    ):
        """Return (obs_26d, roll, pitch).  Returns (None, 0, 0) if data not ready.

        commanded_pos / commanded_vel: when provided, substitute for /joint_states
        (which is always-zero on hardware).  Mirrors the sim condition where
        Gazebo joint states ≈ commanded positions under a fast PD controller.
        """
        with self._lock:
            js  = self._js
            imu = self._imu

        if js is None or imu is None:
            return None, 0.0, 0.0

        if commanded_pos is not None:
            pos = np.clip(commanded_pos, -math.pi, math.pi).astype(np.float32)
            vel = (commanded_vel if commanded_vel is not None
                   else np.zeros(12, dtype=np.float32)).astype(np.float32)
        else:
            # Name-indexed lookup — tolerates any joint ordering in the published message
            name_to_i = {n: i for i, n in enumerate(js.name)}
            pos = np.zeros(12, dtype=np.float32)
            vel = np.zeros(12, dtype=np.float32)
            for k, jname in enumerate(JOINT_ORDER):
                j = name_to_i.get(jname)
                if j is not None:
                    pos[k] = float(js.position[j]) if j < len(js.position) else 0.0
                    vel[k] = float(js.velocity[j]) if j < len(js.velocity) else 0.0

        q = imu.orientation
        roll, pitch = _quat_to_rpy(q.x, q.y, q.z, q.w)

        obs = np.concatenate([
            pos, vel,
            np.array([roll, pitch], dtype=np.float32),
        ]).astype(np.float32)

        return obs, float(roll), float(pitch)

    @property
    def js_count(self):
        with self._lock:
            return self._js_count

    @property
    def imu_count(self):
        with self._lock:
            return self._imu_count


# ---------------------------------------------------------------------------
# Main deployer
# ---------------------------------------------------------------------------

class PolicyDeployer:

    def __init__(self, args):
        self._args = args

        rospy.init_node('deploy_policy', anonymous=False, disable_signals=True)

        self._obs  = _ObsCollector()
        self._pub  = rospy.Publisher(
            '/ros_robot_controller/robot/set_joint_angle',
            SetJointAngle,
            queue_size=1,
        )

        # ONNX session
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            args.policy,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        rospy.loginfo(f"[deploy] Loaded policy: {args.policy}")

        self._running = True
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, *_):
        self._running = False

    # ------------------------------------------------------------------
    # Joint command helpers
    # ------------------------------------------------------------------

    def _sim_to_servo(self, sim_angles: np.ndarray) -> np.ndarray:
        """Convert sim-space joint angles (rad) to servo units (0–1000)."""
        servo = sim_angles * _SIM_SIGN * _SIM_SCALE + _SIM_OFFSET
        return np.clip(servo, 0.0, 1000.0).astype(np.float32)

    def _publish_targets(self, sim_targets: np.ndarray, duration: float) -> None:
        """Convert sim angles → servo units → SDK format, then publish."""
        servo  = self._sim_to_servo(sim_targets)
        sdk_flat = servo[_POLICY_TO_SDK].tolist()
        msg = SetJointAngle()
        msg.joint_angle = sdk_flat
        msg.duration    = float(duration)
        self._pub.publish(msg)

    def _go_to_stand(self, duration: float = 1.0) -> None:
        """Command all joints to hardware natural standing pose."""
        self._publish_targets(_STAND_OFFSET.copy(), duration)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, obs: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
        """Run ONNX inference.  Returns scaled action clipped to ±GAIT_RESIDUAL."""
        inp = obs[np.newaxis].astype(np.float32)
        raw = self._sess.run([self._output_name], {self._input_name: inp})[0][0]
        return (np.clip(raw, -GAIT_RESIDUAL, GAIT_RESIDUAL) * action_scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        args = self._args

        # ── wait for topic data ──────────────────────────────────────────
        rospy.loginfo("[deploy] Waiting for /joint_states and /imu ...")
        if not self._obs.wait_for_data(timeout=10.0):
            rospy.logerr("[deploy] Timed out waiting for sensor data — is bringup running?")
            return

        rospy.loginfo(
            "[deploy] Topics ready.  /joint_states: %d msgs, /imu: %d msgs",
            self._obs.js_count, self._obs.imu_count,
        )

        # ── initial stand ────────────────────────────────────────────────
        if not args.no_init:
            rospy.loginfo("[deploy] Commanding stand pose (2 s) ...")
            self._go_to_stand(duration=2.0)
            rospy.sleep(2.5)

            # Verify robot is upright before starting
            obs, roll, pitch = self._obs.build_obs()
            if obs is None:
                rospy.logerr("[deploy] No obs after stand — aborting")
                return
            if abs(roll) > FALL_THRESH or abs(pitch) > FALL_THRESH:
                rospy.logwarn(
                    "[deploy] Robot not upright after stand (roll=%.2f, pitch=%.2f) — aborting",
                    roll, pitch,
                )
                return
            rospy.loginfo("[deploy] Stand OK (roll=%.2f rad, pitch=%.2f rad)", roll, pitch)

        if args.dry_run:
            rospy.loginfo("[deploy] --dry-run: printing obs/action, NOT publishing commands")

        # ── control loop ─────────────────────────────────────────────────
        hz        = float(args.hz)
        dt        = 1.0 / hz
        max_delta = args.max_vel * dt    # max joint movement per step (rad)
        servo_dur = args.servo_duration
        # Hardware thigh servos are opposite to Gazebo URDF direction; negate when set.
        direction = -1.0 if args.invert_gait else 1.0
        rate      = rospy.Rate(hz)

        prev_targets  = _STAND_OFFSET.copy()  # start clamping relative to standing, not zeros
        prev_vel      = np.zeros(12, dtype=np.float32)  # finite-diff velocity for commanded obs
        t_start       = time.time()
        step          = 0
        falls         = 0

        rospy.loginfo(
            "[deploy] Starting policy loop: hz=%.0f  max_seconds=%.0f  dry_run=%s",
            hz, args.max_seconds, args.dry_run,
        )

        while self._running:
            t_now   = time.time()
            elapsed = t_now - t_start

            if elapsed >= args.max_seconds:
                rospy.loginfo("[deploy] Max time reached (%.1f s).", elapsed)
                break

            if args.use_commanded_obs:
                # Subtract _STAND_OFFSET so obs[0:12] is zero-centred, matching
                # the Gazebo training distribution (URDF stand = all zeros).
                # Multiply by direction so obs polarity matches training (+ve = forward).
                obs, roll, pitch = self._obs.build_obs(
                    commanded_pos=direction * (prev_targets - _STAND_OFFSET),
                    commanded_vel=direction * prev_vel,
                )
            else:
                obs, roll, pitch = self._obs.build_obs()
            if obs is None:
                rospy.logwarn_throttle(2.0, "[deploy] Waiting for obs ...")
                rate.sleep()
                continue

            # Safety watchdog — stop immediately if robot has fallen
            if abs(roll) > FALL_THRESH or abs(pitch) > FALL_THRESH:
                falls += 1
                rospy.logwarn(
                    "[deploy] FALL DETECTED: roll=%.2f  pitch=%.2f — stopping", roll, pitch
                )
                self._go_to_stand(duration=0.5)
                break

            # Thigh gait + RL residual (direction-sensitive) + calf lift (direction-invariant).
            gait_time   = elapsed
            gait_ref    = _gait_targets(gait_time)
            calf_ref    = _calf_lift(gait_time, args.calf_amp)
            action      = self._infer(obs, action_scale=args.action_scale)
            sim_targets = direction * (gait_ref + action) + calf_ref + _STAND_OFFSET

            # Velocity clamp in sim-angle space
            delta       = np.clip(sim_targets - prev_targets, -max_delta, max_delta)
            sim_targets = prev_targets + delta

            if args.dry_run:
                if step % int(hz) == 0:  # print once per second
                    rospy.loginfo(
                        "[dry-run] t=%.1f  roll=%.3f  pitch=%.3f  "
                        "action_rms=%.4f  thigh_rf_cmd=%.3f  calf_rf_cmd=%.3f  thigh_rf_obs=%.3f",
                        elapsed, roll, pitch,
                        float(np.sqrt(np.mean(action ** 2))),
                        float(sim_targets[1]),           # rf_thigh cmd
                        float(sim_targets[2]),           # rf_calf cmd (oscillates co-phased with thigh)
                        float(obs[1]),                   # rf_thigh obs (should be ~0 at standing)
                    )
            else:
                self._publish_targets(sim_targets, duration=servo_dur)

            prev_vel     = (sim_targets - prev_targets) / dt
            prev_targets = sim_targets.copy()
            step        += 1
            rate.sleep()

        # ── safe stop ────────────────────────────────────────────────────
        elapsed_total = time.time() - t_start
        rospy.loginfo(
            "[deploy] Loop ended: %d steps  %.1f s  %d falls",
            step, elapsed_total, falls,
        )
        if not args.dry_run:
            rospy.loginfo("[deploy] Sending stand pose ...")
            self._go_to_stand(duration=1.0)
            rospy.sleep(1.5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Deploy an ONNX RL walking policy on the real ROSPug.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--policy', required=True,
        help='Path to the .onnx policy file (export with export_onnx.py).',
    )
    p.add_argument(
        '--hz', type=float, default=50.0,
        help='Control loop rate in Hz.  Must match training rate (50 Hz) for correct policy behaviour.',
    )
    p.add_argument(
        '--max-seconds', type=float, default=10.0,
        help='Hard stop after this many seconds.',
    )
    p.add_argument(
        '--no-init', action='store_true',
        help='Skip the initial stand command (use if robot is already standing).',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Subscribe and run inference but do NOT publish joint commands.',
    )
    p.add_argument(
        '--use-commanded-obs', action='store_true',
        help=(
            'Use the previous commanded joint angles as obs[0:24] instead of '
            '/joint_states (which is always-zero on hardware — no encoder readback). '
            'Matches the sim condition: Gazebo joint states ≈ commanded positions.'
        ),
    )
    p.add_argument(
        '--invert-gait', action='store_true',
        help='Negate the gait reference and RL action before commanding servos. '
             'Use when hardware thigh direction is opposite to the Gazebo URDF '
             '(robot walks backward without this flag).',
    )
    p.add_argument(
        '--action-scale', type=float, default=0.3,
        help='Scale factor applied to the RL residual action (0–1). '
             'Lower values reduce servo stress and wild motion. Default 0.3.',
    )
    p.add_argument(
        '--max-vel', type=float, default=2.0,
        help='Maximum joint velocity in rad/s for the per-step velocity clamp. '
             'Default 2.0 (was 4.0 — halved to protect RF servo).',
    )
    p.add_argument(
        '--servo-duration', type=float, default=0.05,
        help='Duration (s) passed to set_joint_angle for each command. '
             'Larger values smooth servo motion and reduce mechanical stress. '
             'Default 0.05 s (was dt=0.02 s).',
    )
    p.add_argument(
        '--calf-amp', type=float, default=0.10,
        help='Calf foot-lift amplitude in rad. Half-wave (swing phase only). '
             '0 = thigh-only gait. Default 0.10 rad.',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.policy):
        sys.stderr.write(f"[FATAL] Policy file not found: {args.policy}\n")
        sys.exit(1)

    deployer = PolicyDeployer(args)
    try:
        deployer.run()
    finally:
        rospy.signal_shutdown('deploy_policy exiting')


if __name__ == '__main__':
    main()
