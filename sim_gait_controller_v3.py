#!/usr/bin/env python3
"""
sim_gait_controller_v3.py  (v3 - joint-space trot + yaw + strafe)
Joint-space trot gait controller for ROSPug in Gazebo.

Extends v2 (forward/backward only) with:
  - Yaw (turning): differential sinusoidal thigh swing -- right legs swing
    at a larger amplitude than left legs (or vice versa), creating turns
    even when vx = 0.  Replaces the static yaw_bias offset used in v2.
  - Strafe (lateral): sinusoidal hip abduction joints driven by vy.
    Hip joints rotate about +-X (longitudinal axis) in base_link -- these
    are the lateral abduction joints.  Driving them in phase with the trot
    cycle makes feet step sideways during swing and return during stance.

URDF kinematic analysis (pug.urdf.xacro):
  Standing pose = all Gazebo joints at 0.
  Thigh/calf joints rotate about base_link +-Y:
    Right legs (RF, RB): axis [0,-1,0]  ->  positive angle = foot FORWARD
    Left  legs (LF, LB): axis [0,+1,0]  ->  positive angle = foot BACKWARD
  Hip joints rotate about base_link +-X (longitudinal):
    RF, RB: axis [-1,0,0]  ->  positive angle = leg swings RIGHT (-Y)
    LF, LB: axis [+1,0,0]  ->  positive angle = leg swings LEFT  (+Y)
  THIGH_SIGN = CALF_SIGN = {rf:+1,rb:+1,lf:-1,lb:-1} normalises sagittal mirror.
  HIP_STRAFE_SIGN        = {rf:-1,rb:-1,lf:+1,lb:+1} normalises lateral mirror.

IMPORTANT -- always restart Gazebo before testing:
    roslaunch pug_description gazebo.launch   (then press Play)
Then in a second container shell:
    python3 /root/rospug_research/sim_gait_controller_v3.py

Movement commands:
    rostopic pub /cmd_vel geometry_msgs/Twist '{linear:  {x:  0.08}}' -r 10   # forward
    rostopic pub /cmd_vel geometry_msgs/Twist '{linear:  {x: -0.08}}' -r 10   # backward
    rostopic pub /cmd_vel geometry_msgs/Twist '{angular: {z:  0.3}}'  -r 10   # turn left
    rostopic pub /cmd_vel geometry_msgs/Twist '{angular: {z: -0.3}}'  -r 10   # turn right
    rostopic pub /cmd_vel geometry_msgs/Twist '{linear:  {y:  0.06}}' -r 10   # strafe left
    rostopic pub /cmd_vel geometry_msgs/Twist '{linear:  {y: -0.06}}' -r 10   # strafe right

Tuning if direction is inverted:
    Strafe reversed  -> flip all HIP_STRAFE_SIGN values (rf/rb: +1, lf/lb: -1)
    Turn  reversed   -> negate the angular.z value in the command
"""

import math
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
THIGH_AMP       = 0.30   # thigh swing amplitude at vx = 1.0 m/s    [rad]
CALF_AMP        = 0.20   # calf foot-lift amplitude during swing      [rad]
YAW_SWING_SCALE = 0.25   # yaw rate -> differential thigh amplitude   [rad/(rad/s)]
HIP_AMP         = 0.25   # hip swing amplitude at vy = 1.0 m/s        [rad]
GAIT_FREQ       = 1.5    # trot cycles per second                     [Hz]
CTRL_RATE       = 50     # controller publish rate                    [Hz]

# ---------------------------------------------------------------------------
# Per-leg kinematics constants  (from URDF joint frame analysis)
# ---------------------------------------------------------------------------
# Right-leg thighs rotate about [0,-1,0]: positive alpha = foot forward.
# Left-leg  thighs rotate about [0,+1,0]: positive alpha = foot backward.
THIGH_SIGN   = {'rf': +1, 'rb': +1, 'lf': -1, 'lb': -1}
CALF_SIGN    = {'rf': +1, 'rb': +1, 'lf': -1, 'lb': -1}

# Hip abduction joints rotate about +-X (longitudinal axis) in base_link.
# RF/RB: positive angle -> leg swings RIGHT (-Y).
# LF/LB: positive angle -> leg swings LEFT  (+Y).
# For strafing LEFT (vy > 0): RF/RB need negative hip angle, LF/LB positive.
HIP_STRAFE_SIGN = {'rf': -1, 'rb': -1, 'lf': +1, 'lb': +1}

# Trot diagonal pairs: RF+LB (phase 0) and LF+RB (phase 0.5).
PHASE_OFFSET = {'rf': 0.0, 'lb': 0.0, 'lf': 0.5, 'rb': 0.5}


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------
class SimGaitController:

    def __init__(self):
        rospy.init_node('sim_gait_controller_v3', anonymous=False)

        # Publisher dict: leg -> (hip_pub, thigh_pub, calf_pub)
        self._pubs = {}
        for leg in ('rf', 'lf', 'rb', 'lb'):
            self._pubs[leg] = (
                rospy.Publisher(
                    '/pug/{}_joint_position_controller/command'.format(leg),
                    Float64, queue_size=1),
                rospy.Publisher(
                    '/pug/{}_thigh_position_controller/command'.format(leg),
                    Float64, queue_size=1),
                rospy.Publisher(
                    '/pug/{}_calf_position_controller/command'.format(leg),
                    Float64, queue_size=1),
            )

        self._vx  = 0.0
        self._vy  = 0.0
        self._yaw = 0.0
        rospy.Subscriber('/cmd_vel', Twist, self._cmd_vel_cb)

        self._phase = 0.0
        self._dt    = 1.0 / CTRL_RATE

        rospy.loginfo('sim_gait_controller_v3: standing up ...')
        self._go_to_stand()
        rospy.loginfo('sim_gait_controller_v3: ready. Publish /cmd_vel to move.')

    # ------------------------------------------------------------------
    def _cmd_vel_cb(self, msg):
        self._vx  = msg.linear.x
        self._vy  = msg.linear.y
        self._yaw = msg.angular.z

    # ------------------------------------------------------------------
    def _publish(self, leg, hip, thigh, calf):
        h, t, c = self._pubs[leg]
        h.publish(Float64(hip))
        t.publish(Float64(thigh))
        c.publish(Float64(calf))

    # ------------------------------------------------------------------
    def _go_to_stand(self):
        """Send all joints to 0 (standing pose) and hold for 1 s."""
        rate = rospy.Rate(CTRL_RATE)
        rospy.sleep(0.4)                  # wait for subscribers to connect
        for _ in range(CTRL_RATE):        # 1 second at standing pose
            for leg in self._pubs:
                self._publish(leg, 0.0, 0.0, 0.0)
            rate.sleep()
            if rospy.is_shutdown():
                return

    # ------------------------------------------------------------------
    def _leg_angles(self, leg, gait_phase):
        """
        Joint-space trot angles for one leg.

        Thigh (sagittal, forward/backward):
          total_amp = THIGH_AMP*vx  +/-  YAW_SWING_SCALE*|yaw|*yaw_dir
          Right legs (+) get larger amplitude when yaw > 0 (turn left);
          left legs (-) get smaller amplitude -> differential stride = turn.
          sinusoidal: thigh = THIGH_SIGN * total_amp * sin(phase)

        Hip (lateral, strafe):
          hip = HIP_STRAFE_SIGN * HIP_AMP * vy * sin(phase)
          Same phase as thigh so foot lifts and steps sideways simultaneously.

        Calf (foot lift during swing only):
          calf = CALF_SIGN * CALF_AMP * max(0, sin(phase))
        """
        p    = (gait_phase + PHASE_OFFSET[leg]) % 1.0
        sinp = math.sin(2.0 * math.pi * p)

        # --- Thigh: forward/backward swing + yaw differential ---
        yaw_dir = math.copysign(1.0, self._yaw) if abs(self._yaw) > 0.01 else 0.0
        if leg in ('rf', 'rb'):
            total_amp = THIGH_AMP * self._vx + YAW_SWING_SCALE * abs(self._yaw) * yaw_dir
        else:
            total_amp = THIGH_AMP * self._vx - YAW_SWING_SCALE * abs(self._yaw) * yaw_dir
        thigh = THIGH_SIGN[leg] * total_amp * sinp

        # --- Hip: lateral swing for strafing ---
        hip = HIP_STRAFE_SIGN[leg] * HIP_AMP * self._vy * sinp

        # --- Calf: foot lift during swing only ---
        moving   = abs(self._vx) > 0.01 or abs(self._vy) > 0.01 or abs(self._yaw) > 0.01
        lift_amp = CALF_AMP if moving else 0.0
        calf     = CALF_SIGN[leg] * lift_amp * max(0.0, sinp)

        return hip, thigh, calf

    # ------------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            speed = math.sqrt(self._vx**2 + self._vy**2) + abs(self._yaw)
            if speed > 0.01:
                self._phase = (self._phase + GAIT_FREQ * self._dt) % 1.0
            # When stopped, phase frozen -> joints hold position

            for leg in self._pubs:
                hip, thigh, calf = self._leg_angles(leg, self._phase)
                self._publish(leg, hip, thigh, calf)

            rate.sleep()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        SimGaitController().run()
    except rospy.ROSInterruptException:
        pass
