"""オイラー角から各回転表現への変換
"""
import os
import sys
import math

import numpy as np

from type_hint import *

from euler_state import EulerState

def euler_to_rot(theta1_deg: float, 
                 theta2_deg: float, 
                 theta3_deg: float, 
                 euler_state: EulerState) -> np.ndarray:
    """オイラー角から回転行列を求める

    Args:
        theta1_deg (float): 第一オイラー角
        theta2_deg (float): 第二オイラー角
        theta3_deg (float): 第三オイラー角
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: 回転行列[3x3]
    """

    return euler_state.to_rot(theta1_deg, theta2_deg, theta3_deg)


def rot_to_euler(rot: np.ndarray,
                 euler_state: EulerState) -> Tuple[float, float, float]:
    """回転行列からオイラー角求める

    Args:
        rot (np.ndarray): 回転行列[3x3]
        euler_state (EulerState): オイラー角の定義

    Returns:
        Tuple[float, float, float]: オイラー角(θ1,θ2,θ3)
    """
    return euler_state.from_rot(rot)

