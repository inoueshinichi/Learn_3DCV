"""回転行列
"""

import os
import sys
import math

import numpy as np

from type_hint import *


def axis_x(rot: np.ndarray) -> np.ndarray:
    """回転行列からX軸ベクトルを抽出

    Args:
        rot (np.ndarray): 回転行列[3x3]

    Returns:
        np.ndarray: X軸方向ベクトル[3x1]
    """
    if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    norm = np.linalg.norm(rot[:,0]) # X軸成分
    nx = rot[:,0] / norm # X軸の方向ベクトル(単位ベクトル)
    return nx

def axis_y(rot: np.ndarray) -> np.ndarray:
    """回転行列からY軸ベクトルを抽出

    Args:
        rot (np.ndarray): 回転行列[3x3]

    Returns:
        np.ndarray: Y軸方向ベクトル[3x1]
    """
    if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    norm = np.linalg.norm(rot[:,1]) # Y軸成分
    ny = rot[:,1] / norm # Y軸の方向ベクトル(単位ベクトル)
    return ny

def axis_z(rot: np.ndarray) -> np.ndarray:
    """回転行列からZ軸ベクトルを抽出

    Args:
        rot (np.ndarray): 回転行列[3x3]

    Returns:
        np.ndarray: Z軸方向ベクトル[3x1]
    """
    if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    norm = np.linalg.norm(rot[:,2]) # Z軸成分
    nz = rot[:,2] / norm # Z軸の方向ベクトル(単位ベクトル)
    return nz

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
    if rot.shape() != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    A = 0.5 * (rot - rot.T)
    l = np.array([A[2,1],A[0,2],A[1,0]], dtype=np.float32) # [nx*sin,ny*sin,nz*sin]
    s = np.linalg.norm(l) # sin
    c = 0.5 * (rot[0,0] + rot[1,1] + rot[2,2] - 1) # cos

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
        theta = math.arctan2(s, c)
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
    if rot.shape() != (3,3):
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
        qx = k2 * math.sign(r32-r23)
        qy = k3 * math.sign(qx*(r21+r12))
        qz = k4 * math.sign(qx*(r31+r13))
    elif k3 >= k2 and k3 >= k4: 
        # max:k3
        qy = k3 * math.sign(r13-r31)
        qx = k2 * math.sign(qy*(r21+r12))
        qz = k4 * math.sign(qy*(r32+r23))
    else:
        # max:k4
        qz = k4 * math.sign(r21-r12)
        qx = k2 * math.sign(qz*(r13+r31))
        qy = k3 * math.sign(qz*(r23+r32))
    
    # quat = [qw,qx,qy,qz] = qw + qx*i + qy*j + qz*k
    return np.array([qw,qx,qy,qz], dtype=np.float32)

