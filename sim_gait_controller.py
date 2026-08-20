#!/usr/bin/env python3
"""
sim_gait_controller.py  (v2 - joint-space trot)
Joint-space trot gait controller for ROSPug in Gazebo.

Bypasses the hardware-dependent ros_robot_controller / STM32 pipeline entirely.
Publishes Float64 commands directly to the 12 effort_controllers/JointPositionController
topics that Gazebo listens to.

URDF kinematic analysis (pug.urdf.xacro):
  Standing pose = all Gazebo joints at 0 (Gazebo physics default).
  Thigh and calf joints both rotate about base_link +-Y axis:
    Right legs (RF, RB): axis [0,-1,0]  ->  positive angle = foot FORWARD
    Left  legs (LF, LB): axis [0,+1,0]  ->  positive angle = foot BACKWARD
  Positive calf = foot UP for right legs, foot DOWN for left legs.
  THIGH_SIGN = CALF_SIGN = {rf:+1,rb:+1,lf:-1,lb:-1} normalises all legs.
  Trot diagonal pairs: RF+LB swing together, LF+RB swing together.

Launch Gazebo first:
    roslaunch pug_description gazebo.launch   (then press Play)
Then in a second container shell:
    python3 /root/rospug_research/sim_gait_controller.py
Send movement commands:
    rostopic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.05}}' -r 10
    rostopic pub /cmd_vel geometry_msgs/Twist '{angular: {z: 0.3}}' -r 10
"""

import math
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
THIGH_AMP  = 0.30   # thigh swing amplitude at vx = 1.0 m/s   [rad]
CALF_AMP   = 0.20   # calf foot-lift amplitude during swing     [rad]
YAW_SCALE  = 0.25   # yaw rate -> thigh differential            [rad/(rad/s)]
GAIT_FREQ  = 1.5    # trot cycles per second                    [Hz]
CTRL_RATE  = 50     # controller publish rate                   [Hz]

# ---------------------------------------------------------------------------
# Per-leg kinematics constants  (from URDF joint frame analysis)
# ---------------------------------------------------------------------------
# Right-leg thighs rotate about [0,-1,0]: positive alpha = foot forward.
# Left-leg  thighs rotate about [0,+1,0]: positive alpha = foot backward.
# Flip sign so the same gait formula drives all four legs correctly.
THIGH_SIGN   = {'rf': +1, 'rb': +1, 'lf': -1, 'lb': -1}
CALF_SIGN    = {'rf': +1, 'rb': +1, 'lf': -1, 'lb': -1}

# Trot pairing: RF+LB (phase 0) and LF+RB (phase 0.5).
PHASE_OFFSET = {'rf': 0.0, 'lb': 0.0, 'lf': 0.5, 'rb': 0.5}



# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------
class SimGaitController:

    def __init__(self):
        rospy.init_node('sim_gait_controller', anonymous=False)

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

        rospy.loginfo('sim_gait_controller: standing up ...')
        self._go_to_stand()
        rospy.loginfo('sim_gait_controller: ready. Publish /cmd_vel to move.')

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
        """
        Send all joints to 0 (Gazebo natural standing pose) and hold for 1 s.
        Short sleep before publishing ensures publisher connections are up.
        """
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

        thigh = SIGN * amp * sin(2*pi*(phase + offset))
            positive half-cycle -> foot swings FORWARD  (swing phase)
            negative half-cycle -> foot sweeps BACKWARD (stance, propels body)

        calf = SIGN * lift * max(0, sin(...))
            only positive during swing -> lifts foot off ground

        THIGH_SIGN / CALF_SIGN correct for left/right kinematic mirror.
        """
        p    = (gait_phase + PHASE_OFFSET[leg]) % 1.0
        sinp = math.sin(2.0 * math.pi * p)

        # Thigh swing proportional to forward speed
        thigh_amp = THIGH_AMP * self._vx

        # Yaw differential: right legs +, left legs -
        if leg in ('rf', 'rb'):
            yaw_bias = self._yaw * YAW_SCALE
        else:
            yaw_bias = -self._yaw * YAW_SCALE

        thigh = THIGH_SIGN[leg] * thigh_amp * sinp + yaw_bias

        # Foot lift only during swing (sinp > 0)
        moving   = abs(self._vx) > 0.01 or abs(self._yaw) > 0.01
        lift_amp = CALF_AMP if moving else 0.0
        calf     = CALF_SIGN[leg] * lift_amp * max(0.0, sinp)

        return 0.0, thigh, calf   # hip kept at neutral

    # ------------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(CTRL_RATE)
        while not rospy.is_shutdown():
            speed = math.sqrt(self._vx**2 + self._vy**2) + abs(self._yaw)
            if speed > 0.01:
                self._phase = (self._phase + GAIT_FREQ * self._dt) % 1.0
            # When stopped, phase is frozen -> joints hold their position

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
