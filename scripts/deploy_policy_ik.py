#!/usr/bin/env python3
"""
deploy_policy_ik.py  v2 — Deploy RL walking policy via firmware Cartesian IK.

v1 failure root cause: the 2D planar IK was inaccurate outside the calibration
point because the ROSPug URDF geometry is non-planar and complex.

v2 fix: use the firmware's own IK service (/ros_robot_controller/robot/set_leg_ik)
which computes exact servo values that correctly account for the full URDF geometry.
A gait lookup table is pre-computed once at startup (< 1 s), then the control
loop runs at 50 Hz with pure table-lookup — no IK at runtime.

Architecture:
  Cartesian foot trajectory  (stance: +X→-X sweep; swing: raised-cosine Z arc)
      ↓  /ros_robot_controller/robot/set_leg_ik  (firmware IK — exact)
      ↓  servo values 0–1000, N_GAIT-sample lookup table (built at startup)
  + RL residual action  (optional; --action-scale 0 = pure IK gait for first test)
      ↓  /ros_robot_controller/robot/set_joint_angle  @ 50 Hz

Run ON THE ROBOT via SSH:

  ssh -t hiwonder@192.168.123.1 \\
    "source /opt/ros/melodic/setup.zsh && source ~/pug/devel/setup.zsh && \\
     PYTHONPATH=/opt/ros/melodic/lib/python3/dist-packages:\\$PYTHONPATH \\
     python3 ~/deploy_policy_ik.py --policy ~/policy_C_500k.onnx \\
       --no-init --max-seconds 10 --action-scale 0"

SDK foot coordinate frame (origin = thigh joint, i.e. hip abduction joint):
  X: robot forward (+), backward (-)
  Y: lateral  — fixed at standing value from get_leg_position (preserves hip angle)
  Z: up (+), down (-)  — ground contact ≈ -0.13 m at standing
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
    from ros_robot_controller.srv import SetLegIK, GetLegPosition
except ImportError:
    sys.stderr.write(
        "[FATAL] Cannot import ros_robot_controller messages/services.\n"
        "  Source ~/pug/devel/setup.zsh and set PYTHONPATH correctly.\n"
    )
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    sys.stderr.write("[FATAL] onnxruntime not found.  pip3 install onnxruntime\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Policy joint ordering (matches rospug_env.py JOINT_ORDER)
JOINT_ORDER = (
    'rf_joint', 'rf_thigh', 'rf_calf',
    'lf_joint', 'lf_thigh', 'lf_calf',
    'rb_joint', 'rb_thigh', 'rb_calf',
    'lb_joint', 'lb_thigh', 'lb_calf',
)

GAIT_FREQ     = 1.5    # Hz — diagonal trot frequency
GAIT_RESIDUAL = 0.15   # rad — RL residual action clip
N_GAIT        = 100    # samples pre-computed per gait cycle

STANCE_FRAC   = 0.5    # fraction of gait cycle in stance (symmetric trot)

# Phase biases in SDK ordering [FR, FL, BR, BL]  (FR=rf, FL=lf, BR=rb, BL=lb)
# FR+BL swing together (phase 0.5); FL+BR swing together (phase 0.0 = stance start)
_PHASE_BIAS_SDK = np.array([0.5, 0.0, 0.0, 0.5])

_DEF_STEP_HALF = 0.020  # m — half-stride length per leg
_DEF_Z_CLEAR   = 0.025  # m — swing clearance (peak height above ground)

FALL_THRESH   = 0.7    # rad

# ---------------------------------------------------------------------------
# Sim-angle ↔ servo conversion  (JOINT_ORDER = policy ordering)
# Verbatim from deploy_policy.py — do NOT change independently.
# ---------------------------------------------------------------------------
_SIM_SCALE  = 1000.0 / math.pi  # servo units per radian ≈ 318.31
_SIM_SIGN   = np.array([  1,   1,  -1,   1,   1,   1,   1,   1,   1,   1,   1,  -1], dtype=np.float32)
_SIM_OFFSET = np.array([600, 250, 500, 400, 750, 500, 600, 250, 500, 400, 750, 500], dtype=np.float32)

# Standing pose in sim-angle space (calibrated to go_home hardware pose)
_STAND_OFFSET = np.array([
    -0.239,  0.200,  0.0,   # rf_joint, rf_thigh, rf_calf
     0.239, -0.200,  0.0,   # lf_joint, lf_thigh, lf_calf
    -0.239,  0.200,  0.0,   # rb_joint, rb_thigh, rb_calf
     0.239, -0.200,  0.0,   # lb_joint, lb_thigh, lb_calf
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Index mappings between POLICY ordering and SDK flat ordering
#
# SDK 3x4 flat layout (SetJointAngle.joint_angle / set_leg_ik output):
#   row 0 (calf) : [FR, FL, BR, BL] -> flat indices [0,1,2,3]
#   row 1 (thigh): [FR, FL, BR, BL] -> flat indices [4,5,6,7]
#   row 2 (hip)  : [FR, FL, BR, BL] -> flat indices [8,9,10,11]
#
# POLICY ordering -> SDK flat:
#   SDK[0..3]  calf  <- policy[2,5,8,11]
#   SDK[4..7]  thigh <- policy[1,4,7,10]
#   SDK[8..11] hip   <- policy[0,3,6,9]
# ---------------------------------------------------------------------------
_POLICY_TO_SDK = np.array([2, 5, 8, 11, 1, 4, 7, 10, 0, 3, 6, 9], dtype=np.intp)

# Inverse mapping: SDK flat -> POLICY ordering
_SDK_TO_POLICY = np.zeros(12, dtype=np.intp)
_SDK_TO_POLICY[_POLICY_TO_SDK] = np.arange(12, dtype=np.intp)

# Standing servo values in SDK flat ordering
_STAND_SERVO_SDK = np.clip(
    (_STAND_OFFSET * _SIM_SIGN * _SIM_SCALE + _SIM_OFFSET)[_POLICY_TO_SDK],
    0.0, 1000.0,
).astype(np.float32)


# ---------------------------------------------------------------------------
# Foot trajectory (SDK coordinate frame, thigh joint origin)
# ---------------------------------------------------------------------------

def _gait_foot_pos_sdk(p_global, cx4, cy4, cz4, step_half, z_clear):
    """Return 3x4 foot position matrix for global phase p_global.

    SDK ordering: [FR, FL, BR, BL].
    X = robot forward (+), Z = down (negative at ground contact).
    Y = lateral; kept fixed (cy4) to preserve hip abduction angle.

    Stance: foot sweeps from center_x+step_half -> center_x-step_half
            (body advances over a forward-placed, fixed ground contact).
    Swing:  foot arcs from center_x-step_half -> center_x+step_half
            with raised-cosine Z clearance (smooth at both endpoints).
    """
    x4 = np.array(cx4, dtype=np.float64)
    y4 = np.array(cy4, dtype=np.float64)
    z4 = np.array(cz4, dtype=np.float64)
    for i in range(4):
        p = (p_global + _PHASE_BIAS_SDK[i]) % 1.0
        if p < STANCE_FRAC:
            ps = p / STANCE_FRAC          # 0 -> 1 during stance
            x4[i] = cx4[i] + step_half * (1.0 - 2.0 * ps)   # front -> back
            # z4[i] unchanged -- ground contact height
        else:
            pw = (p - STANCE_FRAC) / (1.0 - STANCE_FRAC)    # 0 -> 1 during swing
            x4[i] = cx4[i] + step_half * (2.0 * pw - 1.0)   # back -> front
            z4[i] = cz4[i] + z_clear * (1.0 - math.cos(2.0 * math.pi * pw)) / 2.0
    return np.array([x4, y4, z4])


# ---------------------------------------------------------------------------
# IMU helper
# ---------------------------------------------------------------------------

def _quat_to_rpy(qx, qy, qz, qw):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


# ---------------------------------------------------------------------------
# Thread-safe observation collector
# ---------------------------------------------------------------------------

class _ObsCollector:
    """Subscribes to /joint_states and /imu; builds 26-D obs on demand."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._js        = None
        self._imu       = None
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

    def wait_for_data(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._js is not None and self._imu is not None:
                    return True
            time.sleep(0.1)
        return False

    def build_obs(self, commanded_pos=None, commanded_vel=None):
        """Return (obs_26d, roll, pitch) or (None, 0, 0) if data not ready."""
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
        rospy.init_node('deploy_policy_ik', anonymous=False, disable_signals=True)

        self._obs = _ObsCollector()
        self._pub = rospy.Publisher(
            '/ros_robot_controller/robot/set_joint_angle',
            SetJointAngle,
            queue_size=1,
        )

        # Firmware IK service proxies
        rospy.loginfo("[deploy_ik] Waiting for IK services ...")
        rospy.wait_for_service('/ros_robot_controller/robot/set_leg_ik',      timeout=15.0)
        rospy.wait_for_service('/ros_robot_controller/robot/get_leg_position', timeout=15.0)
        self._set_ik_srv  = rospy.ServiceProxy('/ros_robot_controller/robot/set_leg_ik',      SetLegIK)
        self._get_pos_srv = rospy.ServiceProxy('/ros_robot_controller/robot/get_leg_position', GetLegPosition)
        rospy.loginfo("[deploy_ik] IK services ready.")

        # Load ONNX policy
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            args.policy,
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )
        self._in_name  = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name
        rospy.loginfo("[deploy_ik] Policy loaded: %s", args.policy)

        self._running = True
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, *_):
        self._running = False

    # ------------------------------------------------------------------
    # Servo publishing helpers
    # ------------------------------------------------------------------

    def _publish_servo(self, servo_sdk_flat, duration):
        """Publish 12-element SDK-ordered servo array to hardware."""
        msg = SetJointAngle()
        msg.joint_angle = np.clip(servo_sdk_flat, 0.0, 1000.0).tolist()
        msg.duration    = float(duration)
        self._pub.publish(msg)

    def _go_to_stand(self, duration=1.0):
        self._publish_servo(_STAND_SERVO_SDK, duration)

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def _infer(self, obs, action_scale=1.0):
        inp = obs[np.newaxis].astype(np.float32)
        raw = self._sess.run([self._out_name], {self._in_name: inp})[0][0]
        return (np.clip(raw, -GAIT_RESIDUAL, GAIT_RESIDUAL) * action_scale).astype(np.float32)

    # ------------------------------------------------------------------
    # Gait table pre-computation (uses firmware IK)
    # ------------------------------------------------------------------

    def _precompute_gait(self, cx4, cy4, cz4, step_half, z_clear):
        """Build N_GAIT-sample servo table using firmware IK.

        Returns array shape (N_GAIT, 12), SDK flat ordering, servo units 0-1000.
        Calls set_leg_ik once per sample — no IK computation in the control loop.
        """
        rospy.loginfo(
            "[deploy_ik] Pre-computing gait table (%d samples, "
            "step_half=%.3f m, z_clear=%.3f m) ...",
            N_GAIT, step_half, z_clear,
        )
        table = np.zeros((N_GAIT, 12), dtype=np.float32)
        for k in range(N_GAIT):
            p  = k / float(N_GAIT)
            fp = _gait_foot_pos_sdk(p, cx4, cy4, cz4, step_half, z_clear)
            resp = self._set_ik_srv(fp.reshape(12).tolist())
            if not resp.success:
                raise RuntimeError(
                    "set_leg_ik failed at sample k={} fp={}".format(k, fp)
                )
            table[k] = np.array(resp.joint_angle, dtype=np.float32)
        rospy.loginfo("[deploy_ik] Gait table ready.")
        return table

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self):
        args = self._args

        # Wait for sensor topics
        rospy.loginfo("[deploy_ik] Waiting for /joint_states and /imu ...")
        if not self._obs.wait_for_data(timeout=10.0):
            rospy.logerr("[deploy_ik] Sensor data timeout — is bringup running?")
            return
        rospy.loginfo(
            "[deploy_ik] Sensors ready: /joint_states %d msgs, /imu %d msgs",
            self._obs.js_count, self._obs.imu_count,
        )

        # Initial stand
        if not args.no_init:
            rospy.loginfo("[deploy_ik] Commanding stand pose (2 s) ...")
            self._go_to_stand(duration=2.0)
            rospy.sleep(2.5)
            _, roll, pitch = self._obs.build_obs()
            if abs(roll) > FALL_THRESH or abs(pitch) > FALL_THRESH:
                rospy.logwarn(
                    "[deploy_ik] Not upright after stand (roll=%.2f pitch=%.2f) — aborting",
                    roll, pitch,
                )
                return
            rospy.loginfo("[deploy_ik] Stand OK (roll=%.2f pitch=%.2f)", roll, pitch)

        # Read current foot positions — used as gait centers.
        # Using the actual Y values preserves hip abduction angle during the gait.
        rospy.loginfo("[deploy_ik] Reading standing foot positions ...")
        resp = self._get_pos_srv()
        if not resp.success:
            rospy.logerr("[deploy_ik] get_leg_position failed — aborting")
            return
        fp_stand = np.array(resp.position, dtype=np.float64).reshape(3, 4)
        cx4, cy4, cz4 = fp_stand[0].copy(), fp_stand[1].copy(), fp_stand[2].copy()
        rospy.loginfo(
            "[deploy_ik] Foot centers  X:[%.3f %.3f %.3f %.3f]  "
            "Y:[%.3f %.3f %.3f %.3f]  Z:[%.3f %.3f %.3f %.3f]",
            cx4[0], cx4[1], cx4[2], cx4[3],
            cy4[0], cy4[1], cy4[2], cy4[3],
            cz4[0], cz4[1], cz4[2], cz4[3],
        )

        # Pre-compute gait table
        try:
            gait_table = self._precompute_gait(cx4, cy4, cz4, args.step_half, args.z_clear)
        except Exception as exc:
            rospy.logerr("[deploy_ik] Gait pre-compute failed: %s", exc)
            return

        # Control loop setup
        hz            = float(args.hz)
        dt            = 1.0 / hz
        max_delta_srv = args.max_vel * dt * _SIM_SCALE  # servo units/step
        servo_dur     = args.servo_duration
        rate          = rospy.Rate(hz)

        stand_servo = _STAND_SERVO_SDK.copy()
        prev_servo  = stand_servo.copy()
        t_start     = time.time()
        step        = 0

        if args.dry_run:
            rospy.loginfo("[deploy_ik] --dry-run: inference active, commands suppressed.")

        rospy.loginfo(
            "[deploy_ik] Starting loop: hz=%.0f  max_seconds=%.0f  "
            "action_scale=%.2f  ramp=%.1f s",
            hz, args.max_seconds, args.action_scale, args.ramp_time,
        )

        while self._running:
            t_now   = time.time()
            elapsed = t_now - t_start

            if elapsed >= args.max_seconds:
                rospy.loginfo("[deploy_ik] Time limit (%.1f s).", elapsed)
                break

            # Build observation
            if args.use_commanded_obs:
                # Convert previous servo (SDK flat) -> sim_angle (policy ordering)
                sim_cmd = (prev_servo[_SDK_TO_POLICY] - _SIM_OFFSET) / (_SIM_SIGN * _SIM_SCALE)
                obs, roll, pitch = self._obs.build_obs(commanded_pos=sim_cmd - _STAND_OFFSET)
            else:
                obs, roll, pitch = self._obs.build_obs()

            if obs is None:
                rospy.logwarn_throttle(2.0, "[deploy_ik] Waiting for obs ...")
                rate.sleep()
                continue

            # Fall watchdog
            if abs(roll) > FALL_THRESH or abs(pitch) > FALL_THRESH:
                rospy.logwarn("[deploy_ik] FALL: roll=%.2f pitch=%.2f — stopping", roll, pitch)
                if not args.dry_run:
                    self._go_to_stand(0.5)
                break

            # Amplitude ramp (0 -> 1 over ramp_time seconds)
            ramp = min(1.0, elapsed / args.ramp_time) if args.ramp_time > 0.0 else 1.0

            # Gait table lookup
            p_global   = (GAIT_FREQ * elapsed) % 1.0
            k          = int(p_global * N_GAIT) % N_GAIT
            servo_gait = gait_table[k]

            # Smooth ramp: blend from standing to full gait
            servo_base = stand_servo + ramp * (servo_gait - stand_servo)

            # RL residual correction (optional)
            if args.action_scale > 0.0:
                action = self._infer(obs, action_scale=args.action_scale)
                # action is in policy ordering, sim_angle units
                # -> servo delta in SDK flat ordering
                servo_delta = (action * _SIM_SIGN * _SIM_SCALE)[_POLICY_TO_SDK]
                servo_cmd   = servo_base + servo_delta
            else:
                servo_cmd = servo_base.copy()

            # Velocity clamp in servo space
            delta     = np.clip(servo_cmd - prev_servo, -max_delta_srv, max_delta_srv)
            servo_cmd = prev_servo + delta

            if not args.dry_run:
                self._publish_servo(servo_cmd, servo_dur)

            prev_servo = servo_cmd.copy()
            step      += 1

            if step % int(hz) == 0:
                rospy.loginfo(
                    "[deploy_ik] t=%.1f s  ramp=%.2f  roll=%.3f  pitch=%.3f  "
                    "phase=%.2f  k=%d",
                    elapsed, ramp, roll, pitch, p_global, k,
                )

            rate.sleep()

        # Safe stop
        rospy.loginfo("[deploy_ik] Loop ended: %d steps  %.1f s", step, time.time() - t_start)
        if not args.dry_run:
            rospy.loginfo("[deploy_ik] Sending stand pose ...")
            self._go_to_stand(duration=1.0)
            rospy.sleep(1.5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Deploy ONNX RL walking policy on ROSPug using firmware IK gait.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--policy',         required=True,
                   help='Path to .onnx policy file.')
    p.add_argument('--hz',             type=float, default=50.0,
                   help='Control loop rate (Hz).')
    p.add_argument('--max-seconds',    type=float, default=10.0,
                   help='Hard time limit.')
    p.add_argument('--step-half',      type=float, default=_DEF_STEP_HALF,
                   help='Half-stride in metres (foot sweeps +/-step_half in X). '
                        'Safe range 0.010-0.030 m.')
    p.add_argument('--z-clear',        type=float, default=_DEF_Z_CLEAR,
                   help='Swing foot clearance in metres. Safe range 0.010-0.040 m.')
    p.add_argument('--no-init',        action='store_true',
                   help='Skip initial stand command (robot already standing).')
    p.add_argument('--dry-run',        action='store_true',
                   help='Run inference but do NOT publish commands.')
    p.add_argument('--use-commanded-obs', action='store_true',
                   help='Use commanded joint angles as obs[0:24] '
                        '(workaround for always-zero /joint_states on hardware).')
    p.add_argument('--action-scale',   type=float, default=0.0,
                   help='RL residual scale (0=pure IK gait). Start at 0, then try 0.3.')
    p.add_argument('--ramp-time',      type=float, default=1.5,
                   help='Seconds to ramp step amplitude from 0 to full.')
    p.add_argument('--max-vel',        type=float, default=8.0,
                   help='Servo velocity clamp (rad/s equivalent).')
    p.add_argument('--servo-duration', type=float, default=0.05,
                   help='Duration parameter passed to set_joint_angle (s).')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.policy):
        sys.stderr.write("[FATAL] Policy file not found: %s\n" % args.policy)
        sys.exit(1)

    deployer = PolicyDeployer(args)
    try:
        deployer.run()
    finally:
        rospy.signal_shutdown('deploy_policy_ik exiting')


if __name__ == '__main__':
    main()

