"""クォータニオン
    q = [qx, qy, qz, qw]
    q = qx*i + qy*j + qz*k + qw

    i -> j -> k -> i
    i*i = -1, j*j = -1, k*k = -1
    i*j = k, j*k = i, k*j = i
    j*i = -k, i*k = -j, k*j = -i

    回転を表現するクォータニオンは単位クォータニオンである必要がある.
    |q| = 1 (qx*qx + qy*qy + qz*qz + qw*qw = 1)
"""

import os
import sys
import math

import numpy as np

from type_hint import *

def slerp(q: np.ndarray, p: np.ndarray, f: float) -> np.ndarray:
    """クォータニオンの球面補完

    Args:
        q (np.ndarray): 始点クォータニオン[4x1]
        p (np.ndarray): 終点クォータニオン[4x1]
        f (float): 内分点比率 0~1

    Returns:
        np.ndarray: 球面補完クォータニオン[4x1]
    """
    pass

def dt_quat(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """クォータニオンの時間微分

    Args:
        q (np.ndarray): クォータニオン[4x1]
        omega (np.ndarray): 角速度ベクトル[3x1] (rad/s)

        dq/dt = - 1/2 * cat_quat([omega, 0] * quat)

    Returns:
        np.ndarray: クォータニオンの時間微分[4x1]
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    if omega.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {q.shape}")
    
    return -0.5 * cat_quat(np.r_[omega, 0], q)

def norm_quat(q: np.ndarray) -> float:
    """クォータニオンのノルム

    Args:
        q (np.ndarray): クォータニオン[4x1] (qx,qy,qz,qw)

        |q|^2 = qx^2 + qy^2 + qz^2 + qw^2
        |q| = sqrt(qx^2 + qy^2 + qz^2 + qw^2)

    Returns:
        float: クォータニオンのノルム
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    return np.linalg.norm(q)

def normalize_quat(q: np.ndarray) -> np.ndarray:
    """クォータニオンの正規化

    Args:
        q (np.ndarray): 非単位クォータニオン[4x1]

        nq = q / |q|

    Returns:
        np.ndarray: 単位クォータニオン[4x1]
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    return q / norm_quat(q)

def iden_quat() -> np.ndarray:
    """単位元クォータニオン

    Returns:
        np.ndarray: (0,0,0,1)
    """
    return np.array([0,0,0,1], dtype=np.float32)

def conj_quat(q: np.ndarray) -> np.ndarray:
    """共役クォータニオン

    Args:
        q (np.ndarray): クォータニオン[4x1] (qx,qy,qz,qw)

    Returns:
        np.ndarray: 共役クォータニオン[4x1] (-qx,-qy,-qz,qw)
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

def inv_quat(q: np.ndarray) -> np.ndarray:
    """逆クォータニオン

    Args:
        q (np.ndarray): クォータニオン[4x1]

        q_inv = conj_q / |q|^2

    Returns:
        np.ndarray: 逆クォータニオン[4x1]
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    return conj_quat(q) / (norm_quat(q) ** 2)

def cat_quat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """クォータニオンのグラスマン積
    ※pにqを結合する. q*p

    Args:
        p (np.ndarray): クォータニオン(px, py, pz, pw) = [p_v, p_s]
        q (np.ndarray): クォータニオン(qx, qy, qz, qw) = [q_v, q_s]

    グラスマン積
    (q*p)_v = q_s*p_v + p_s*q_v + cross(q_v,p_v)
    (q*p)_s = q_s*p_s - dot(q_v,p_v)

    q*p = [
        qx*pw+qw*px-qz*py+qy*pz, # (q*p)_vx
        qy*pw+qz*px+qw*py-qx*pz, # (q*p)_vy
        qz*pw-qy*px+qx*py+qw*pz, # (q*p)_vz
        qw*pw-qx*px-qy*py-qz*pz, # (q*p)_s
    ]

    Returns:
        np.ndarray: クォータニオン[4x1]
    """
    px, py, pz, pw = p[0], p[1], p[2], p[3]
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    # グラスマン積
    qp_x = qx*pw+qw*px-qz*py+qy*pz
    qp_y = qy*pw+qz*px+qw*py-qx*pz
    qp_z = qz*pw-qy*px+qx*py+qw*pz
    qp_w = qw*pw-qx*px-qy*py-qz*pz
    qp = np.array([qp_x, qp_y, qp_z, qp_w], dtype=np.float32)
    
    # 連結したクォータニオンが単位クォータニオンであるとは限らない
    return normalize_quat(qp)

def rot_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """点(位置ベクトル)をクォータニオンで回転する

    Args:
        v (np.ndarray): 点(位置ベクトル)[3x1]
        q (np.ndarray): 単位クォータニオン[4x1]

    Returns:
        np.ndarray: 回転後の点(位置ベクトル)
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    if norm_quat(q) != 1.0:
        raise ValueError(f"Not match norm 1.0. Given is {norm_quat(q)}")
    
    if v.shape() != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {v.shape}")
    
    r = np.r_[v, 0]     # 点をクォータニオンに変換
    q_inv = inv_quat(q) # 逆クォータニオン

    rq = cat_quat(q_inv, cat_quat(r, q))
    ret_v = np.array([rq[0], rq[1], rq[2]], dtype=np.float32)
    return ret_v


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """クォータニオンから回転行列を求める

    Args:
        q (np.ndarray): クォータニオン[4x1] (qx,qy,qz,qw)

        rot = 
        [[1-2*qy^2-2*qz^2, 2*qx*qy+2*qw*qz, 2*qx*qz-2*qw*qy],
         [2*qx*qy-2*qw*qz, 1-2*qx^2-2*qz^2, 2*qy*qz+2*qw*qx],
         [2*qx*qz+2*qw*qy, 2*qy*qz-2*qw*qx, 1-2*qx^2-2*qy^2]]

    Returns:
        np.ndarray: 回転行列[3x3]
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    if norm_quat(q) != 1.0:
        raise ValueError(f"Not match norm 1.0. Given is {norm_quat(q)}")
    
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    
    r11 = 1-2*qy**2-2*qz**2
    r12 = 2*qx*qy+2*qw*qz
    r13 = 2*qx*qz-2*qw*qy
    r21 = 2*qx*qy-2*qw*qz
    r22 = 1-2*qx^2-2*qz^2
    r23 = 2*qy*qz+2*qw*qx
    r31 = 2*qx*qz+2*qw*qy
    r32 = 2*qy*qz-2*qw*qx
    r33 = 1-2*qx**2-2*qy**2

    rot = np.array([
        [r11,r12,r13],
        [r21,r22,r23],
        [r31,r32,r33]
    ], dtype=np.float32)

    return rot

def quat_to_rvec(q: np.ndarray) -> np.ndarray:
    """クォータニオンから回転ベクトルを求める

    Args:
        q (np.ndarray): クォータニオン[4x1]

    Returns:
        np.ndarray: 回転ベクトル[3x1]
    """
    if q.shape() != (4,1):
        raise ValueError(f"Not match shape (4,1). Given is {q.shape}")
    
    if norm_quat(q) != 1.0:
        raise ValueError(f"Not match norm 1.0. Given is {norm_quat(q)}")
    
    theta = 2 * math.acos(q[3])
    nx, ny, nz = 0.0, 0.0, 0.0

    if theta == 0.0:
        # 無回転 (qw=1)
        pass
    elif theta == math.pi:
        # 180°回転 (qw=0)
        nx, ny, nz = q[0], q[1], q[2]
    else:
        s2 = math.sin(theta/2)
        nx, ny, nz = q[0]/s2, q[1]/s2, q[2]/s2

    return theta * np.array([nx, ny, nz], dtype=np.float32)

