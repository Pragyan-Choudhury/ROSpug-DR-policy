#!/usr/bin/env python3.8
"""
obs_bridge_test.py — Verify the 26D RL observation vector can be assembled
from real-hardware ROS topics before deploying a policy.

Topics read:
  /joint_states              — 12 joint positions + velocities (joint_state_publisher)
  /imu                       — filtered orientation (imu_complementary_filter)
  /ros_robot_controller/battery — voltage check

Run INSIDE the real-robot Docker container:
  python3.8 /root/rospug_research/scripts/obs_bridge_test.py [--seconds 15]

Exit codes:
  0 — observation vector verified OK
  1 — required topic unavailable or data looks wrong
"""

import sys
import math
import time
import argparse
import threading
from typing import Optional, List

import rospy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import UInt16

# Joint order must match rospug_env.py exactly
JOINT_ORDER: tuple = (
    'rf_joint', 'rf_thigh', 'rf_calf',
    'lf_joint', 'lf_thigh', 'lf_calf',
    'rb_joint', 'rb_thigh', 'rb_calf',
    'lb_joint', 'lb_thigh', 'lb_calf',
)

# Plausibility bounds for a standing robot (radians)
JOINT_POS_WARN  = 2.0    # |pos| > this → suspicious (servo overrange)
JOINT_VEL_WARN  = 10.0   # |vel| > this → suspicious (noise / bad data)
ORIENT_WARN     = 0.3    # |roll| or |pitch| > this when standing → suspicious


def _quat_to_rpy(qx: float, qy: float, qz: float, qw: float):
    """Same inline RPY conversion as rospug_env.py — no tf dependency."""
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


class ObsBridgeTester:
    def __init__(self):
        self._lock = threading.Lock()

        self._js_msg: Optional[JointState] = None
        self._imu_msg: Optional[Imu] = None
        self._battery_mv: Optional[int] = None

        self._js_count = 0
        self._imu_count = 0

        rospy.Subscriber('/joint_states', JointState, self._js_cb, queue_size=5)
        rospy.Subscriber('/imu', Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber('/ros_robot_controller/battery', UInt16, self._bat_cb, queue_size=1)

    def _js_cb(self, msg: JointState):
        with self._lock:
            self._js_msg = msg
            self._js_count += 1

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._imu_msg = msg
            self._imu_count += 1

    def _bat_cb(self, msg: UInt16):
        with self._lock:
            self._battery_mv = msg.data

    # ── observation construction (mirrors rospug_env.py exactly) ─────────
    def build_obs(self):
        """Return (obs_26d, roll, pitch, ok) where ok=False means bad data."""
        with self._lock:
            if self._js_msg is None or self._imu_msg is None:
                return None, 0.0, 0.0, False

            js = self._js_msg
            imu = self._imu_msg

        # ── Joint positions and velocities ────────────────────────────────
        name_to_idx = {name: i for i, name in enumerate(js.name)}
        pos = [0.0] * 12
        vel = [0.0] * 12
        missing = []
        for out_idx, jname in enumerate(JOINT_ORDER):
            if jname in name_to_idx:
                src = name_to_idx[jname]
                pos[out_idx] = js.position[src] if js.position else 0.0
                vel[out_idx] = js.velocity[src] if js.velocity else 0.0
            else:
                missing.append(jname)

        # ── Body orientation ─────────────────────────────────────────────
        q = imu.orientation
        qmag = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        imu_valid = qmag > 0.5   # magnitude near 1 means real quaternion

        if imu_valid:
            roll, pitch = _quat_to_rpy(q.x, q.y, q.z, q.w)
        else:
            # /imu quaternion is zero — fall back to accelerometer-only estimate
            ax = imu.linear_acceleration.x
            ay = imu.linear_acceleration.y
            az = imu.linear_acceleration.z
            if abs(az) > 0.1:
                roll  = math.atan2(ay, az)
                pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
            else:
                roll, pitch = 0.0, 0.0

        import numpy as np
        obs = np.array(pos + vel + [roll, pitch], dtype=np.float32)
        ok = len(missing) == 0
        return obs, roll, pitch, ok

    # ── diagnostics ───────────────────────────────────────────────────────
    def run(self, duration: float):
        print("\n" + "=" * 60)
        print("  obs_bridge_test.py — Real-hardware observation check")
        print(f"  ROS master : {rospy.get_param('/run_id', '?')}")
        print("=" * 60)

        # Wait for topics
        print("\n[1/3] Waiting for /joint_states and /imu (up to 10 s) ...")
        deadline = time.time() + 10.0
        while time.time() < deadline:
            with self._lock:
                got_js  = self._js_msg is not None
                got_imu = self._imu_msg is not None
            if got_js and got_imu:
                break
            time.sleep(0.2)

        with self._lock:
            got_js  = self._js_msg is not None
            got_imu = self._imu_msg is not None

        if not got_js:
            print("  [FAIL] /joint_states — no data received in 10 s")
            print("         Is pug_bringup/base.launch running on the robot?")
            return 1
        print("  [OK]   /joint_states received")

        if not got_imu:
            print("  [WARN] /imu — no data.  Orientation will use accelerometer fallback.")
            print("         Check that imu_complementary_filter is running.")
        else:
            print("  [OK]   /imu received")

        with self._lock:
            bat = self._battery_mv
        if bat is not None:
            status = "OK" if bat > 7400 else "LOW"
            print(f"  [BAT]  Battery: {bat} mV  ({status})")
        else:
            print("  [WARN] /ros_robot_controller/battery not received")

        # Print joint name mapping
        print("\n[2/3] Joint name order verification ...")
        with self._lock:
            js_names = list(self._js_msg.name) if self._js_msg else []
        print(f"  /joint_states names  : {js_names}")
        print(f"  Required JOINT_ORDER : {list(JOINT_ORDER)}")
        for jname in JOINT_ORDER:
            if jname not in js_names:
                print(f"  [WARN] '{jname}' missing from /joint_states")
        missing_count = sum(1 for j in JOINT_ORDER if j not in js_names)
        if missing_count == 0:
            print("  [OK]  All 12 joints present")
        else:
            print(f"  [WARN] {missing_count} joints missing — zeros used for those slots")

        # Live 26D observation stream
        print(f"\n[3/3] Live 26D observation stream for {duration:.0f} s ...")
        print("  Format: obs[0:12]=pos, obs[12:24]=vel, obs[24]=roll, obs[25]=pitch")
        print()

        t0 = time.time()
        sample = 0
        all_ok = True
        while time.time() - t0 < duration and not rospy.is_shutdown():
            obs, roll, pitch, ok = self.build_obs()
            if obs is None:
                time.sleep(0.1)
                continue

            sample += 1
            if sample % 25 == 1:   # print header every 25 samples (0.5 Hz)
                print(f"  {'t(s)':>5} {'roll°':>7} {'pit°':>7}  "
                      f"{'pos_range':>12}  {'vel_range':>12}  {'status':>6}")

            pos_arr = obs[0:12]
            vel_arr = obs[12:24]
            status = "OK"
            if any(abs(p) > JOINT_POS_WARN for p in pos_arr):
                status = "WARN:pos"
                all_ok = False
            if any(abs(v) > JOINT_VEL_WARN for v in vel_arr):
                status = "WARN:vel"
                all_ok = False
            if abs(roll) > ORIENT_WARN or abs(pitch) > ORIENT_WARN:
                status = "WARN:orient"
                all_ok = False

            t_elapsed = time.time() - t0
            print(f"  {t_elapsed:5.1f}  "
                  f"{math.degrees(roll):7.2f}  {math.degrees(pitch):7.2f}  "
                  f"  [{min(pos_arr):+.3f},{max(pos_arr):+.3f}]  "
                  f"  [{min(vel_arr):+.3f},{max(vel_arr):+.3f}]  "
                  f"  {status}")

            time.sleep(0.5)

        with self._lock:
            js_hz  = self._js_count / max(duration, 1.0)
            imu_hz = self._imu_count / max(duration, 1.0)

        print()
        print("=" * 60)
        print(f"  /joint_states rate : {js_hz:.1f} Hz  (expected ≥ 50 Hz)")
        print(f"  /imu rate          : {imu_hz:.1f} Hz  (expected ≥ 50 Hz)")
        if all_ok:
            print("  [PASS] Observation vector looks correct — ready for deploy_policy.py")
        else:
            print("  [WARN] Some values outside normal bounds — review output above")
        print("=" * 60 + "\n")

        return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Verify real-hardware observation bridge")
    parser.add_argument('--seconds', type=float, default=10.0,
                        help='How long to stream live observations (default: 10)')
    args = parser.parse_args()

    rospy.init_node('obs_bridge_test', anonymous=True)
    tester = ObsBridgeTester()
    rc = tester.run(args.seconds)
    sys.exit(rc)


if __name__ == '__main__':
    main()
