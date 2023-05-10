"""回転ベクトル
V = (rx, ry, rz)
方向ベクトル N = V/|V| = (nx, ny, nz), θ = |V|
"""

import os
import sys
import math

import numpy as np

from type_hint import *

def make_rvec(n: np.ndarray, theta: float) -> np.ndarray:
    """回転行列の計算

    Args:
        n (np.ndarray): 方向ベクトル(単位ベクトル)
        theta (float): 回転角 [rad]

    Returns:
        np.ndarray: 回転ベクトル[3x1]
    """
    if n.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {n.shape}")

    n /= np.linalg.norm(n)
        
    return theta * n

def rot_with_rvec(v: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルによる点の回転

    Args:
        v (np.ndarray): 点群(複数のいちベクトル) [3xN]
        rvec (np.ndarray): 回転行列[3x1]

    Returns:
        np.ndarray: 回転後の点群[3xN]
    """
    if rvec.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rvec.shape}")
    
    if v.shape[0] != 3:
        raise ValueError(f"Not match shape (3,N). Given is {v.shape}")

    theta = np.linalg.norm(rvec) # rad
    c = math.cos(theta)
    s = math.sin(theta)
    n = rvec / theta # 方向ベクトル [3x1]

    # 方向ベクトルの歪対称行列
    Nx = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ], dtype=np.float32) # [3x3]

    # 回転ベクトルによる点群の回転
    return v*c + n * (n @ v) * (1 - c) + Nx @ v * s

     
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
         [nz*nx{1-c(θ)}+ny*s(θ), nz*ny{1-c(θ)}+nx*s(θ), nz*nz{1-c(θ)}+c(θ)]]

        単位行列[3x3] I
        直交行列?[3x3] nn = n*n.T
        歪対称行列[3x3] Sn[n|x] 各要素は方向ベクトル
        R = c(θ) * I + {1 - c(θ)} * nn + s(θ) * Sn

    Returns:
        np.ndarray: [3x3]回転行列
    """
    if rvec.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転角[-pi, pi]
    n = rvec / theta # 方向ベクトル (3,1)
    nx, ny, nz = n[0], n[1], n[2]
    # print("theta", theta)
    # print("n", n)

    c = math.cos(theta) # cos
    s = math.sin(theta) # sin

    # バグがある
    # I = np.ones((3,3), dtype=np.float32)
    # nn = n @ n.T # (3,3)
    # print("nn", nn)

    # Sn = np.zeros((3,3), dtype=np.float32)
    # Sn[0,1] = -n[2] # -nz
    # Sn[0,2] = n[1]  # ny
    # Sn[1,0] = n[2]  # nz
    # Sn[1,2] = -n[0] # -nx
    # Sn[2,0] = -n[1] # ny
    # Sn[2,1] = n[0]  # nx
    # print("Sn", Sn)

    # # 回転行列
    # rot = c * I + (1 - c) * nn + s * Sn
    # print("rot", rot)

    # こっちなら正常
    rot = np.zeros((3,3), dtype=np.float32)
    rot[0,0] = nx * nx * (1 - c) + c
    rot[0,1] = nx * ny * (1 - c) - nz * s
    rot[0,2] = nx * nz * (1 - c) + ny * s
    rot[1,0] = ny * nx * (1 - c) + nz * s
    rot[1,1] = ny * ny * (1 - c) + c
    rot[1,2] = ny * nz * (1 - c) - nx * s
    rot[2,0] = nz * nx * (1 - c) + ny * s
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
    if rvec.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rvec.shape}")
    
    theta = np.linalg.norm(rvec) # 回転量
    n = rvec / theta # 方向ベクトル (単位ベクトル)

    s2 = math.sin(theta/2)
    c2 = math.cos(theta/2)
    qx, qy, qz = n[0]*s2, n[1]*s2, n[2]*s2
    qw = c2

    return np.array([qx, qy, qz, qw], dtype=np.float32)


