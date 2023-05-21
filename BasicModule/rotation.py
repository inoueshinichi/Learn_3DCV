"""回転行列
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from BasicModule.euler_state import EulerState

from BasicModule.utility import ax_rot, ay_rot, az_rot

from type_hint import *


def rot_to_euler(rot: np.ndarray,
                 euler_state: EulerState) -> Tuple[float, float, float]:
    """回転行列からオイラー角求める

    Args:
        rot (np.ndarray): 回転行列[3x3]
        euler_state (EulerState): オイラー角の定義

    Returns:
        Tuple[float, float, float]: オイラー角(θ1,θ2,θ3)
    """
    theta1_rad, theta2_rad, theta3_rad = euler_state.from_rot(rot)
    theta1_deg = math.degrees(theta1_rad)
    theta2_deg = math.degrees(theta2_rad)
    theta3_deg = math.degrees(theta3_rad)

    return theta1_deg, theta2_deg, theta3_deg


def rot_to_rvec(rot: np.ndarray) -> np.ndarray:
    """回転行列から回転ベクトルを求める

    Args:
        rot (np.ndarray): 回転行列[3x3]

    通常, 回転行列から2組の回転ベクトルが得られる.
    [V,θ]と[-V,-θ]の2組. 
    しなしながら, 0<=θ<=πと制約を掛けることで,
    回転ベクトルを一意に決めることができる

    Returns:
        np.ndarray: 回転ベクトル[3x1]
    """
    if rot.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    A = 0.5 * (rot - rot.T)
    l = np.array([A[2,1],A[0,2],A[1,0]], dtype=np.float32) # [nx*sin,ny*sin,nz*sin]
    s = np.linalg.norm(l) # sin
    c = 0.5 * (rot[0,0] + rot[1,1] + rot[2,2] - 1) # cos

    # print("A(=0.5*(R-R^T))\n", A)
    # print("sin(θ)=", s)
    # print("cons(θ)=", c)

    rvec = np.array([0, 0, 0], dtype=np.float32) # 回転ベクトル
    n = np.array([0, 0, 0], dtype=np.float32) # 方向ベクトル
    theta = 0.0 # 回転量

    # sin(s)とcos(c)の値で場合分け
    if s == 0 and c == 1:
        # ゼロベクトル
        pass
    elif s == 0 and c == -1:
        # θ=±πが考えられるが, 0=<θ<=πで制約を掛けることで一意に決まる
        theta = math.pi # θ=π
        B = rot + np.eye(3,3, dtype=np.float32) # R + I
        # Bの列ベクトルを取得
        r1, r2, r3 = np.hsplit(B) # B = [r1,r2,r3]
        r_vecs = [np.squeeze(r1), np.squeeze(r2), np.squeeze(r3)]

        # B(R+I)の非ゼロの列ベクトルを取得
        # r1,r2,r3のどれかになる
        non_zero_vec = None
        non_zero_norm = 0.0
        for r_vec in r_vecs:
            norm = np.linalg.norm(r_vec)
            if norm != 0.0:
                non_zero_norm = norm
                non_zero_vec = r_vec
                break
        # 回転量
        theta = math.pi
        # 方向ベクトル
        n = non_zero_vec / non_zero_norm
        # 回転ベクトル
        rvec = theta * n

        # 符号の反転チェック
        r1, r2, r3 = rvec
        
        # 1. Z軸に並行なベクトル
        # 2. YZ平面に存在するベクトル
        # 3. X!=0のベクトル
        if (r1 == 0 and r2 == 0 and r3 < 0) or\
            (r1 == 0 and r2 < 0) or\
            (r1 < 0):
            rvec *= -1

    else: # sin != 0
        # 方向ベクトル
        n = l / s
        # 回転量
        theta = math.atan2(s, c)
        # 回転ベクトル
        rvec = theta * n

    # 出力
    return rvec
        
        
def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """回転行列からクォータニオンを求める

    Args:
        rot (np.ndarray): 回転行列[3x3]

    Returns:
        np.ndarray: クォータニオン[4x1]
    """
    if rot.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    r11, r12, r13 = rot[0,0], rot[0,1], rot[0,2]
    r21, r22, r23 = rot[1,0], rot[1,1], rot[1,2]
    r31, r32, r33 = rot[2,0], rot[2,1], rot[2,2]

    qw, qx, qy, qz = 0.0, 0.0, 0.0, 0.0

    # 対角成分から実部(w)を計算
    tr = r11 + r22 + r33
    qw = 0.5 * math.sqrt(tr + 1)

    # 虚部(x,y,z)の計算
    k2 = 0.5 * math.sqrt(r11-r22-r33+1)
    k3 = 0.5 * math.sqrt(-r11+r22-r33+1)
    k4 = 0.5 * math.sqrt(-r11-r22+r33+1)

    if k2 >= k3 and k2 >= k4: 
        # max:k2
        qx = k2 * np.sign(r32-r23)
        qy = k3 * np.sign(qx*(r21+r12))
        qz = k4 * np.sign(qx*(r31+r13))
    elif k3 >= k2 and k3 >= k4: 
        # max:k3
        qy = k3 * np.sign(r13-r31)
        qx = k2 * np.sign(qy*(r21+r12))
        qz = k4 * np.sign(qy*(r32+r23))
    else:
        # max:k4
        qz = k4 * np.sign(r21-r12)
        qx = k2 * np.sign(qz*(r13+r31))
        qy = k3 * np.sign(qz*(r23+r32))
    
    # quat = [qw,qx,qy,qz] = qw + qx*i + qy*j + qz*k
    quat = np.array([qw,qx,qy,qz], dtype=np.float32)
    return quat / np.linalg.norm(quat)



