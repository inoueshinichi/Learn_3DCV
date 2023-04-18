"""回転ベクトル
V = (rx, ry, rz)
方向ベクトル N = V/|V| = (nx, ny, nz), θ = |V|
"""

import os
import sys
import math

import numpy as np

from type_hint import *

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
    if rvec.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {rotvec.shape}")
    
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

    
