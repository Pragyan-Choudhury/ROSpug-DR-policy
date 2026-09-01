#!/usr/bin/env python3.8
"""
rospug_env_dr.py — Domain-Randomized gymnasium.Env for ROSPug (Step 5).

Subclasses RosPugEnv and resamples three physics parameters every episode:

  1. Robot body mass  — /gazebo/set_link_properties  (pug::base_footprint, ±15%)
  2. Servo latency    — time.sleep() injected in _publish_joints  (0–10 ms)
  3. ODE compliance   — /gazebo/set_physics_properties cfm+erp

Ground-plane mu1/mu2 are NOT randomized: Gazebo 9 has no runtime service to
write per-surface friction.  cfm/erp are used as a surface-compliance proxy.

Prerequisites:
  Same as RosPugEnv — Gazebo running, Play pressed, controllers loaded.

Usage::

    env = RosPugEnvDR()
    obs, info = env.reset()
    print(info['dr_params'])
    # {'mass_frac': 0.932, 'latency_ms': 4.7, 'cfm': 0.000231, 'erp': 0.612}

Step 6 held-out conditions (fix one or more params outside training ranges)::

    light  = RosPugEnvDR(mass_range=(0.80, 0.80))          # −20%, below [0.85,1.15]
    heavy  = RosPugEnvDR(mass_range=(1.20, 1.20))          # +20%, above [0.85,1.15]
    laggy  = RosPugEnvDR(latency_range=(0.015, 0.015))     # 15ms, above [0,10ms]
    slippy = RosPugEnvDR(cfm_range=(0.001, 0.001))         # above [0,0.0005]
    worst  = RosPugEnvDR(mass_range=(1.20,1.20), latency_range=(0.015,0.015))
"""

import time
from typing import Optional, Tuple, Dict, Any

import numpy as np

import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import (
    GetLinkProperties, GetLinkPropertiesRequest,
    SetLinkProperties, SetLinkPropertiesRequest,
    GetPhysicsProperties,
    SetPhysicsProperties, SetPhysicsPropertiesRequest,
)

from rl_env.rospug_env import RosPugEnv


# Gazebo link name: base_link + lidar_link are fixed-joint lumped into base_footprint
_BODY_LINK = 'pug::base_footprint'


class RosPugEnvDR(RosPugEnv):
    """
    RosPugEnv with per-episode domain randomization.

    Parameters
    ----------
    mass_range : (lo, hi) fractions of nominal base_link mass.
        Default (0.85, 1.15) → ±15 % of 0.2454 kg.
    latency_range : (lo, hi) servo command delay, seconds.
        Default (0.0, 0.010) → 0–10 ms.
    cfm_range : ODE constraint-force mixing coefficient.
        0.0 = rigid/hard (Gazebo default); higher = softer ground proxy.
        Default (0.0, 0.0005).
    erp_range : ODE error-reduction parameter.
        Default (0.3, 0.8).
    node_name : rospy node name passed through to RosPugEnv.
    """

    def __init__(
        self,
        mass_range: Tuple[float, float]    = (0.85, 1.15),
        latency_range: Tuple[float, float] = (0.0, 0.010),
        cfm_range: Tuple[float, float]     = (0.0, 0.0005),
        erp_range: Tuple[float, float]     = (0.3, 0.8),
        node_name: str = 'rospug_env_dr',
    ) -> None:
        super().__init__(node_name=node_name)

        self._mass_range    = mass_range
        self._latency_range = latency_range
        self._cfm_range     = cfm_range
        self._erp_range     = erp_range

        # Initialised per reset(); 0.0 before first episode
        self._episode_latency: float = 0.0
        self._dr_params: Dict[str, float] = {}

        # Wire additional Gazebo services
        rospy.wait_for_service('/gazebo/get_link_properties', timeout=15.0)
        self._svc_get_link = rospy.ServiceProxy(
            '/gazebo/get_link_properties', GetLinkProperties)

        rospy.wait_for_service('/gazebo/set_link_properties', timeout=15.0)
        self._svc_set_link = rospy.ServiceProxy(
            '/gazebo/set_link_properties', SetLinkProperties)

        rospy.wait_for_service('/gazebo/get_physics_properties', timeout=15.0)
        self._svc_get_physics = rospy.ServiceProxy(
            '/gazebo/get_physics_properties', GetPhysicsProperties)

        rospy.wait_for_service('/gazebo/set_physics_properties', timeout=15.0)
        self._svc_set_physics = rospy.ServiceProxy(
            '/gazebo/set_physics_properties', SetPhysicsProperties)

        # Snapshot Gazebo's initial ODE config once; used as baseline in every reset
        self._ode_defaults = self._svc_get_physics()

        # Query actual nominal link properties from Gazebo (base_link + lidar_link
        # are fixed-joint-lumped into base_footprint, so mass/inertia/CoM differ
        # from the raw URDF base_link values)
        nom = self._svc_get_link(GetLinkPropertiesRequest(link_name=_BODY_LINK))
        if not nom.success:
            raise RuntimeError(
                f'RosPugEnvDR: could not get link properties for {_BODY_LINK}: '
                f'{nom.status_message}')
        self._nom_mass = nom.mass
        self._nom_com  = nom.com
        self._nom_ixx  = nom.ixx
        self._nom_ixy  = nom.ixy
        self._nom_ixz  = nom.ixz
        self._nom_iyy  = nom.iyy
        self._nom_iyz  = nom.iyz
        self._nom_izz  = nom.izz
        rospy.loginfo(
            'RosPugEnvDR: %s nominal mass=%.4f kg  '
            'CoM=(%.4f, %.4f, %.4f)',
            _BODY_LINK, self._nom_mass,
            self._nom_com.position.x,
            self._nom_com.position.y,
            self._nom_com.position.z,
        )

        rospy.loginfo(
            'RosPugEnvDR ready: mass=[%.2f,%.2f]  latency=[%dms,%dms]  '
            'cfm=[%.4f,%.4f]  erp=[%.2f,%.2f]',
            *mass_range,
            int(latency_range[0] * 1000), int(latency_range[1] * 1000),
            *cfm_range,
            *erp_range,
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Sample new physics params, apply to Gazebo, then delegate to parent."""

        mass_frac = float(np.random.uniform(*self._mass_range))
        latency   = float(np.random.uniform(*self._latency_range))
        cfm       = float(np.random.uniform(*self._cfm_range))
        erp       = float(np.random.uniform(*self._erp_range))

        # Apply before reset_world so params are active for the whole episode.
        # reset_world restores poses/velocities but does NOT revert link properties
        # or ODE physics settings — both persist until we change them again.
        self._set_body_mass(mass_frac)
        self._set_ode_physics(cfm, erp)

        # Store latency so _publish_joints override can use it
        self._episode_latency = latency
        self._dr_params = {
            'mass_frac':  round(mass_frac, 4),
            'latency_ms': round(latency * 1000.0, 2),
            'cfm':        round(cfm, 6),
            'erp':        round(erp, 3),
        }

        rospy.loginfo(
            'DR episode: mass_frac=%.3f  latency=%.1fms  cfm=%.5f  erp=%.3f',
            mass_frac, latency * 1000.0, cfm, erp,
        )

        obs, _ = super().reset(seed=seed, options=options)
        return obs, {'dr_params': self._dr_params}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        info['dr_params'] = self._dr_params
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # _publish_joints override — injects per-episode servo latency
    # ------------------------------------------------------------------

    def _publish_joints(self, targets: np.ndarray) -> None:
        """Delay command by episode latency before passing to parent publisher."""
        if self._episode_latency > 0.0:
            time.sleep(self._episode_latency)
        super()._publish_joints(targets)

    # ------------------------------------------------------------------
    # Physics randomization helpers
    # ------------------------------------------------------------------

    def _set_body_mass(self, mass_frac: float) -> None:
        """
        Scale pug::base_footprint mass and inertia tensor by mass_frac.

        base_footprint is the Gazebo SDF body that results from fixed-joint
        lumping of base_link + lidar_link.  Nominal values are queried from
        Gazebo at __init__ time.  CoM position is preserved exactly.
        """
        req              = SetLinkPropertiesRequest()
        req.link_name    = _BODY_LINK
        req.gravity_mode = True
        req.mass         = self._nom_mass * mass_frac
        req.ixx          = self._nom_ixx  * mass_frac
        req.ixy          = self._nom_ixy  * mass_frac
        req.ixz          = self._nom_ixz  * mass_frac
        req.iyy          = self._nom_iyy  * mass_frac
        req.iyz          = self._nom_iyz  * mass_frac
        req.izz          = self._nom_izz  * mass_frac
        # Preserve original CoM position (offset from link origin due to lumping)
        req.com.position.x    = self._nom_com.position.x
        req.com.position.y    = self._nom_com.position.y
        req.com.position.z    = self._nom_com.position.z
        req.com.orientation.w = 1.0

        resp = self._svc_set_link(req)
        if not resp.success:
            rospy.logwarn('set_link_properties failed: %s', resp.status_message)

    def _set_ode_physics(self, cfm: float, erp: float) -> None:
        """
        Update Gazebo ODE cfm and erp; all other solver fields are restored
        from the snapshot taken at __init__ so they are never accidentally changed.
        """
        d   = self._ode_defaults.ode_config
        req = SetPhysicsPropertiesRequest()
        req.time_step       = self._ode_defaults.time_step
        req.max_update_rate = self._ode_defaults.max_update_rate
        req.gravity         = self._ode_defaults.gravity
        # Populate ode_config field-by-field from defaults, then override cfm/erp
        req.ode_config.auto_disable_bodies         = d.auto_disable_bodies
        req.ode_config.sor_pgs_precon_iters         = d.sor_pgs_precon_iters
        req.ode_config.sor_pgs_iters                = d.sor_pgs_iters
        req.ode_config.sor_pgs_w                    = d.sor_pgs_w
        req.ode_config.sor_pgs_rms_error_tol        = d.sor_pgs_rms_error_tol
        req.ode_config.contact_surface_layer        = d.contact_surface_layer
        req.ode_config.contact_max_correcting_vel   = d.contact_max_correcting_vel
        req.ode_config.max_contacts                 = d.max_contacts
        req.ode_config.cfm                          = cfm
        req.ode_config.erp                          = erp

        resp = self._svc_set_physics(req)
        if not resp.success:
            rospy.logwarn('set_physics_properties failed: %s', resp.status_message)
