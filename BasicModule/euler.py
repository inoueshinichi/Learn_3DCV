"""オイラー角から各回転表現への変換
"""
import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *

from BasicModule.euler_state import EulerState
from BasicModule.rotation import rot_to_rvec, rot_to_quat

def euler_to_rot(theta1_deg: float, 
                 theta2_deg: float, 
                 theta3_deg: float, 
                 euler_state: EulerState) -> np.ndarray:
    """オイラー角から回転行列を求める

    Args:
        theta1_deg (float): 第1オイラー角
        theta2_deg (float): 第2オイラー角
        theta3_deg (float): 第3オイラー角
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: 回転行列[3x3]
    """
    theta1_rad = math.radians(theta1_deg)
    theta2_rad = math.radians(theta2_deg)
    theta3_rad = math.radians(theta3_deg)

    return euler_state.to_rot(theta1_rad, theta2_rad, theta3_rad)


def euler_to_rvec(theta1_deg: float,
                  theta2_deg: float,
                  theta3_deg: float,
                  euler_state: EulerState) -> np.ndarray:
    """オイラー角から回転ベクトルを求める

    Args:
        theta1_deg (float): 第1オイラー角
        theta2_deg (float): 第2オイラー角
        theta3_deg (float): 第3オイラー角
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: 回転ベクトル[3x1]
    """
    # 回転行列
    rot = euler_to_rot(theta1_deg, theta2_deg, theta3_deg , euler_state)

    # 回転ベクトル
    rvec = rot_to_rvec(rot)
    
    return rvec


def euler_to_quat(theta1_deg: float,
                  theta2_deg: float,
                  theta3_deg: float,
                  euler_state: EulerState) -> np.ndarray:
    """オイラー角からクォータニオンを求める

    Args:
        theta1_deg (float): 第1オイラー角
        theta2_deg (float): 第2オイラー角
        theta3_deg (float): 第3オイラー角
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: クォータニオン[4x1]
    """
    # 回転行列
    rot = euler_to_rot(theta1_deg, theta2_deg, theta3_deg , euler_state)

    # クォータニオン
    quat = rot_to_quat(rot)

    return quat





