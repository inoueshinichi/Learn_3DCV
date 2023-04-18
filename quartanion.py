"""クォータニオン
"""

import os
import sys
import math

import numpy as np

from type_hint import *

def quat_norm(q: np.ndarray) -> float:
    """クォータニオンのノルム

    Args:
        q (np.ndarray): クォータニオン[4x1]

    Returns:
        float: クォータニオンのノルム
    """
    if q.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {q.shape}")
    
    return np.linalg.norm(q)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """クォータニオンの正規化

    Args:
        q (np.ndarray): 非単位クォータニオン[4x1]

    Returns:
        np.ndarray: 単位クォータニオン[4x1]
    """
    if q.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {q.shape}")
    
    return q / quat_norm(q)

def quat_conj(q: np.ndarray) -> np.ndarray:
    """共役クォータニオン

    Args:
        q (np.ndarray): クォータニオン[4x1]

    Returns:
        np.ndarray: 共役クォータニオン[4x1]
    """
    if q.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {q.shape}")
    
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def quat_inv(q: np.ndarray) -> np.ndarray:
    """逆クォータニオン

    Args:
        q (np.ndarray): クォータニオン[4x1]

    Returns:
        np.ndarray: 逆クォータニオン[4x1]
    """
    if q.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {q.shape}")
    
    return quat_conj(q) / (quat_norm(q) ** 2)

def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """クォータニオンのグラスマン積

    Args:
        p (np.ndarray): クォータニオン(pw, px, py, pz) = [p_s, p_v]
        q (np.ndarray): クォータニオン(qw, qx, qy, qz) = [q_s, q_v]

    グラスマン積
    (q*p)_v = q_s*p_v + p_s*q_v + cross(q_v,p_v)
    (q*p)_s = q_s*p_s - dot(q_v,p_v)

    q*p = [
        qw*pw-qx*px-qy*py-qz*pz, # (q*p)_s
        qx*pw+qw*px-qz*py+qy*pz, # (q*p)_vx
        qy*pw+qz*px+qw*py-qx*pz, # (q*p)_vy
        qz*pw-qy*px+qx*py+qw*pz  # (q*p)_vz
    ]

    Returns:
        np.ndarray: クォータニオン[4x1]
    """
    pass