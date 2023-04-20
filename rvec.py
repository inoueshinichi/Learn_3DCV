"""回転ベクトル
V = (rx, ry, rz)
方向ベクトル N = V/|V| = (nx, ny, nz), θ = |V|
"""

import os
import sys
import math

import numpy as np

from type_hint import *

def rvec(n: np.ndarray, theta_rad: float) -> np.ndarray:
    """回転行列の計算

    Args:
        n (np.ndarray): 方向ベクトル(単位ベクトル)
        theta_rad (float): 回転量

    Returns:
        np.ndarray: 回転ベクトル[3x1]
    """
    if n.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {n.shape}")
    
    return theta_rad * n

def rvec_rot()
     
def rvec_to_rot(rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルを回転行列に変換(ロドリゲスの回転公式)

    Args:
        rvec (np.ndarray): [3x1]回転ベクトル

        回転ベクトル v = (vx,vy,vz)
        方向ベクトル n = (nx,ny,nz)
        回転量 θ = |v|
        c(θ) = cos(θ)
        s(θ) = sin(θ)
        [[nx*nx{1-c(θ)}+c(θ), nx*ny{1-c(θ)}-nx*s(θ), nx*nz{1-c(θ)}+ny*s(θ)],
         [ny*nx{1-c(θ)}+nz*s(θ), ny*ny{1-c(θ)}+c(θ), ny*nz{1-c(θ)}-nx*s(θ)],
         [nz*nx{1-c(θ)}+ny*s(θ), nz*ny{1-c(θ)}+nx*s(θ), nz*nz{1-c(θ)}+c(θ)]]

        単位行列[3x3] I
        直交行列?[3x3] nn = n*n.T
        歪対称行列[3x3] J[rotvec|x] 各要素は回転ベクトル
        R = c(θ) * I + {1 - c(θ)} * nn + s(θ) * J

    Returns:
        np.ndarray: [3x3]回転行列
    """
    if rvec.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転角[-pi, pi]
    n = rvec / theta # 方向ベクトル

    c = math.cos(theta) # cos
    s = math.sin(theta) # sin
    I = np.ones((3,3), dtype=np.float32)
    nn = n @ n.T
    J = np.zeros((3,3), dtype=np.float32)
    J[0,1] = -rvec[2] # -vz
    J[0,2] = rvec[1]  # vy
    J[1,0] = rvec[2]  # vz
    J[1,2] = -rvec[0] # -vx
    J[2,0] = -rvec[1] # vy
    J[2,1] = rvec[0]  # vx

    # 回転行列
    rot = c * I + (1 - c) * nn + s * J
    
    return rot

def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルからクォータニオンを求める

    Args:
        rvec (np.ndarray): 回転ベクトル[3x1] (vx,vy,vz)

    Returns:
        np.ndarray: クォータニオン[4x1] (qx,qy,qz,qw)
    """
    if rvec.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転量
    n = rvec / theta # 方向ベクトル (単位ベクトル)

    s2 = math.sin(theta/2)
    c2 = math.cos(theta/2)
    qx, qy, qz = n[0]*s2, n[1]*s2, n[2]*s2
    qw = c2

    return np.array([qx, qy, qz, qw], dtype=np.float32)


