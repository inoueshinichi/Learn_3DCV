"""回転ベクトル
V = (rx, ry, rz)
方向ベクトル N = V/|V| = (nx, ny, nz), θ = |V|
"""

import os
import sys
import math

import numpy as np

from type_hint import *

def rotvec_to_rot(rotvec: np.ndarray) -> np.ndarray:
    """回転ベクトルを回転行列に変換(ロドリゲスの回転公式)

    Args:
        rotvector (np.ndarray): [3x1]回転ベクトル

    Returns:
        np.ndarray: [3x3]回転行列
    """
    if rotvec.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rotvec.shape}")
    
    pass

def rotvec_from_rot(rot: np.ndarray) -> np.ndarray:
    """回転行列から回転ベクトルを求める

    Args:
        rot (np.ndarray): [3x3]回転行列

    Returns:
        np.ndarray: [3x1]回転ベクトル
    """
    if rot.shape() != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    pass