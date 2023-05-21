"""回転行列
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *


def ax_rot(ax_rad: float) -> np.ndarray:
    """X軸中心の回転ベクトル

    Args:
        ax_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ax_rot = np.array([
        1, 0, 0,
        0, math.cos(ax_rad), -math.sin(ax_rad),
        0, math.sin(ax_rad), math.cos(ax_rad)
    ], dtype=np.float32).reshape(3,3)
    return ax_rot


def ay_rot(ay_rad: float) -> np.ndarray:
    """Y軸中心の回転ベクトル

    Args:
        ay_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    ay_rot = np.array([
        math.cos(ay_rad), 0, math.sin(ay_rad),
        0, 1, 0,
        -math.sin(ay_rad), 0, math.cos(ay_rad)
    ], dtype=np.float32).reshape(3,3)
    return ay_rot


def az_rot(az_rad: float) -> np.ndarray:
    """Z軸中心の回転行列

    Args:
        az_deg (float): 角度[deg]

    Returns:
        np.ndarray: 回転行列(3x3)
    """
    az_rot = np.array([
        math.cos(az_rad), -math.sin(az_rad), 0,
        math.sin(az_rad), math.cos(az_rad), 0,
        0, 0, 1
    ], dtype=np.float32).reshape(3,3)
    return az_rot


def dt_rot(rot: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """回転行列の微分(ポアソンの微分公式)

    Args:
        rot (np.ndarray): 回転行列[3x3]
        omega (np.ndarray): 角速度ベクトル[3x1]

    Returns:
        np.ndarray: 回転行列の微分[3x3]
    """
    if rot.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")

    if omega.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {omega.shape}")
    
    # 角速度ベクトルの歪対称行列
    tilde_omega: np.ndarray = np.zeros(3, 3, dtype=np.float32)
    tilde_omega[0,1] = omega[2]
    tilde_omega[0,2] = -omega[1]
    tilde_omega[1,0] = -omega[2]
    tilde_omega[1,2] = omega[0]
    tilde_omega[2,0] = omega[1]
    tilde_omega[2,1] = -omega[0]

    return tilde_omega @ rot