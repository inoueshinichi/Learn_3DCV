"""回転行列
"""

import os
import sys
import math

import numpy as np

from type_hint import *

from euler_state import EulerState


def ax_rot(ax_deg:float) -> np.ndarray:
    """X軸中心の回転ベクトル

    Args:
        ax_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ax_rad = math.radians(ax_deg)
    ax_rot = np.array([
        1, 0, 0,
        0, math.cos(ax_rad), -math.sin(ax_rad),
        0, math.sin(ax_rad), math.cos(ax_rad)
    ], dtype=np.float32).reshape(3,3)
    return ax_rot

def ay_rot(ay_deg:float) -> np.ndarray:
    """Y軸中心の回転ベクトル

    Args:
        ay_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ay_rad = math.radians(ay_deg)
    ay_rot = np.array([
        math.cos(ay_rad), 0, math.sin(ay_rad),
        0, 1, 0,
        -math.sin(ay_rad), 0, math.cos(ay_rad)
    ], dtype=np.float32).reshape(3,3)
    return ay_rot

def az_rot(az_deg:float) -> np.ndarray:
    """Z軸中心の回転行列

    Args:
        az_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    az_rad = math.radians(az_deg)
    az_rot = np.array([
        math.cos(az_rad), -math.sin(az_rad), 0,
        math.sin(az_rad), math.cos(az_rad), 0,
        0, 0, 1
    ], dtype=np.float32).reshape(3,3)
    return az_rot

