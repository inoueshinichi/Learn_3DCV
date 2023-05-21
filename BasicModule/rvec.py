"""回転ベクトル
V = (rx, ry, rz)
方向ベクトル N = V/|V| = (nx, ny, nz), θ = |V|
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from BasicModule.euler_state import EulerState

from type_hint import *


def rvec(n: np.ndarray, theta: float) -> np.ndarray:
    """回転ベクトルの計算

    Args:
        n (np.ndarray): 方向ベクトル(単位ベクトル) [3xN]
        theta (float): 回転角 [rad]

    Returns:
        np.ndarray: 回転ベクトル[3xN]
    """
    if n.shape[0] != 3:
        raise ValueError(f"Not match shape (3,) or (3,1). Given is {n.shape}")
    
    return theta * n


def rotate_points_by_rvec(v: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルによる点の回転

    Args:
        v (np.ndarray): 点群(複数のいちベクトル) [3xN]
        rvec (np.ndarray): 回転行列[3x1]

    Returns:
        np.ndarray: 回転後の点群[3xN]
    """
    if v.shape[0] != 3:
        raise ValueError(f"Not match shape (3,N). Given is {v.shape}")

    rot = rvec_to_rot(rvec) # (3,3)

    return rot @ v # 点群の回転

     
def rvec_to_rot(rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルを回転行列に変換(ロドリゲスの回転公式)
    https://w3e.kanazawa-it.ac.jp/math/physics/category/physical_math/linear_algebra/henkan-tex.cgi?target=/math/physics/category/physical_math/linear_algebra/rodrigues_rotation_matrix.html

    Args:
        rvec (np.ndarray): [3x1]回転ベクトル

        回転ベクトル v = (vx,vy,vz)
        方向ベクトル n = (nx,ny,nz)
        回転量 θ = |v|
        c(θ) = cos(θ)
        s(θ) = sin(θ)
        [[nx*nx{1-c(θ)}+c(θ), nx*ny{1-c(θ)}-nz*s(θ), nx*nz{1-c(θ)}+ny*s(θ)],
         [ny*nx{1-c(θ)}+nz*s(θ), ny*ny{1-c(θ)}+c(θ), ny*nz{1-c(θ)}-nx*s(θ)],
         [nz*nx{1-c(θ)}-ny*s(θ), nz*ny{1-c(θ)}+nx*s(θ), nz*nz{1-c(θ)}+c(θ)]]

        単位行列[3x3] I
        直交行列?[3x3] nn = n*n.T
        歪対称行列[3x3] Sn[n|x] 各要素は方向ベクトル
        R = c(θ) * I + {1 - c(θ)} * nn + s(θ) * Sn

    Returns:
        np.ndarray: [3x3]回転行列
    """
    if rvec.shape[0] != 3:
        raise ValueError(f"Not match shape (3,) or (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転角[-pi, pi]
    n = rvec / theta # 方向ベクトル (3,1)
    nx, ny, nz = n[0], n[1], n[2]
   
    c = math.cos(theta) # cos
    s = math.sin(theta) # sin

    rot = np.zeros((3,3), dtype=np.float32)
    rot[0,0] = nx * nx * (1 - c) + c
    rot[0,1] = nx * ny * (1 - c) - nz * s
    rot[0,2] = nx * nz * (1 - c) + ny * s
    rot[1,0] = ny * nx * (1 - c) + nz * s
    rot[1,1] = ny * ny * (1 - c) + c
    rot[1,2] = ny * nz * (1 - c) - nx * s
    rot[2,0] = nz * nx * (1 - c) - ny * s
    rot[2,1] = nz * ny * (1 - c) + nx * s
    rot[2,2] = nz * nz * (1 - c) + c
    
    return rot


def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルからクォータニオンを求める

    Args:
        rvec (np.ndarray): 回転ベクトル[3x1] (vx,vy,vz)

    Returns:
        np.ndarray: クォータニオン[4x1] (qx,qy,qz,qw)
    """
    if rvec.shape[0] != 3:
        raise ValueError(f"Not match shape (3,) or (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転量
    n = rvec / theta # 方向ベクトル (単位ベクトル)

    s2 = math.sin(theta/2)
    c2 = math.cos(theta/2)
    qx, qy, qz = n[0]*s2, n[1]*s2, n[2]*s2
    qw = c2

    return np.array([qx, qy, qz, qw], dtype=np.float32)


def rvec_to_euler(rvec: np.ndarray, euler_state: EulerState) -> Tuple[float, float, float]:
    """回転ベクトルからオイラー角に変換する

    Args:
        rvec (np.ndarray): 回転ベクトル[3x1] (vx,vy,vz)
        euler_state (EulerState): オイラー角の定義

    Returns:
        Tuple[float, float, float]: オイラー角 (θ1,θ2,θ3)
    """
    # 回転ベクトル -> 回転行列
    rot = rvec_to_rot(rvec)

    theta1_rad, theta2_rad, theta3_rad = euler_state.from_rot(rot)
    theta1_deg = math.degrees(theta1_rad)
    theta2_deg = math.degrees(theta2_rad)
    theta3_deg = math.degrees(theta3_rad)

    return theta1_deg, theta2_deg, theta3_deg


