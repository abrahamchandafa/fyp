"""
Thermal Tracker — Infrastructure-Free AR Headset Tracking Using Thermal Sensors.

This package implements a pipeline for projecting, tracking, and estimating 6-DoF
pose from laser-induced thermal dot patterns on surfaces, following the methodology
proposed by Sheinin et al. (CVPR 2024).

Modules
-------
heat_diffusion : Physics-based simulation of heat conduction on surfaces.
pattern_projection : Laser dot pattern generation and projection control.
scene : 3D scene representation and thermal image rendering.
dot_tracker : Detection and frame-to-frame tracking of thermal dots.
pose_estimation : 6-DoF camera pose recovery from 2D–3D correspondences.
visualization : Rendering utilities for thermal frames and trajectories.
"""

__version__ = "0.1.0"
