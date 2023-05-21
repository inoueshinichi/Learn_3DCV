"""3D座標変換
定義: 基点座標系の座標成分(x,y,z,w=1)を対象座標系の座標成分(x',y',z',w'=1)に変換する
@note Transform行列[4x4]
@note inv(Transform行列[4x4]) = Pose行列[4x4]
@note inv(Pose行列[4x4]) = Transform行列[4x4]
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *

from BasicModule.euler_state import EulerState
from BasicModule.rotation import rot_to_quat
from BasicModule.rvec import *
from BasicModule.quartanion import cat_quat, rotate_points_by_quat, quat_to_rot
from BasicModule.euler import euler_to_rot, euler_to_quat
from BasicModule.pose import get_scale, get_trans, get_rot


def transform(t: np.ndarray,
              rot: np.ndarray, 
              scale: np.ndarray) -> np.ndarray:
    """Transform行列[4x4]を作成

    Args:
        t (np.ndarray): 並進要素 [3x1] (tx,ty,tz)
        rot (np.ndarray): 単位クォータニオン[4x1] (qx,qy,qz,qw)
        scale (np.ndarray): スケール要素[3x1] (sx,sy,sz)

    Returns:
        np.ndarray: Transform行列[4x4]
    """
    if t.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {t.shape}")
    if rot.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    if scale.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {scale.shape}")
    
    T = np.zeros((4,4), dtype=np.float32)
    T[:3,:3] = rot # 回転要素(3,3)
    T[:3,3] = t # 並進要素(3,1)
    T[3,3] = 1.0
    T[:3,0] *= scale[0]
    T[:3,1] *= scale[1]
    T[:3,2] *= scale[2]

    return T


def transform(t: np.ndarray,
              quat: np.ndarray,
              scale: np.ndarray) -> np.ndarray:
    """Transform行列[4x4]を作成

    Args:
        t (np.ndarray): 並進要素 [3x1] (tx,ty,tz)
        quat (np.ndarray): 単位クォータニオン[4x1] (qx,qy,qz,qw)
        scale (np.ndarray): スケール要素[3x1] (sx,sy,sz)

    Returns:
        np.ndarray: Transform行列[4x4]
    """
    # クォータニオン -> 回転行列
    rot = quat_to_rot(quat) # (3,3)

    return transform(t=t, rot=rot, scale=scale)


def transform(t: np.ndarray,
              rvec: np.ndarray,
              scale: np.ndarray) -> np.ndarray:
    """Transform行列[4x4]を作成

    Args:
        t (np.ndarray): 並進要素 [3x1] (tx,ty,tz)
        rvec (np.ndarray): 回転ベクトル[3x1] (vx,vy,vz)
        scale (np.ndarray): スケール要素[3x1] (sx,sy,sz)

    Returns:
        np.ndarray: Transform行列[4x4]
    """
    # 回転ベクトル -> 回転行列
    rot = rvec_to_rot(rvec)

    return transform(t=t, rot=rot, scale=scale)


def transform(t: np.ndarray,
              euler_deg: Tuple[float, float, float],
              euler_state: EulerState,
              scale: np.ndarray) -> np.ndarray:
    """Transform行列[4x4]を作成

    Args:
        t (np.ndarray): 並進要素 [3x1] (tx,ty,tz)
        euler_deg (Tuple[float, float, float]): オイラー角(θ1,θ2,θ3)
        euler_state (EulerState): オイラー角の定義
        scale (np.ndarray): スケール要素[3x1] (sx,sy,sz)

    Returns:
        np.ndarray: Transform行列[4x4]
    """
    # オイラー角から回転行列に変換
    theta1_deg, theta2_deg, theta3_deg = euler_deg
    rot = euler_to_rot(theta1_deg=theta1_deg,
                       theta2_deg=theta2_deg,
                       theta3_deg=theta3_deg,
                       euler_state=euler_state)
    inv_rot = rot.T # オイラー角は姿勢を表現する回転行列のため, 座標変換では転置しておく.

    return transform(t=t, rot=inv_rot, scale=scale)


def update_transform(T: np.ndarray,
                     update_t: np.ndarray,
                     update_quat: np.ndarray,
                     update_scale : np.ndarray) -> np.ndarray:
    """Transform行列[4x4]の更新

    Args:
        T (np.ndarray): Pose行列[4x4]
        update_t (np.ndarray): 並進変化[3x1]
        update_quat (np.ndarray): 姿勢(クォータニオン)変化[4x1]
        update_scale (np.ndarray): スケール変化

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """
    if update_t.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {update_t.shape}")
    if update_quat.shape[4] != 4:
        raise ValueError(f"Not match shape (4,1) or (4,). Given is {update_quat.shape}")
    if update_scale.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {update_scale.shape}")

    # 現在の状態
    state_scale = get_scale(T)
    state_rot = get_rot(T)
    state_trans = get_trans(T)
    
    # 回転行列 -> クォータニオン
    state_quat = rot_to_quat(state_rot) # 状態

    # 更新 (回転) : クォータニオンで回転の更新を行う
    new_state_quat = update_quat(state_quat, update_quat) # 姿勢はクォータニオンで更新
    new_state_rot = quat_to_rot(new_state_quat) # 0_R_2 = 0_R_1 @ 1_R_2
    
    # 更新 (その他)
    new_state_trans = state_rot @ update_t + state_trans # 0_T_2 = 0_R_1 @ 1_T_2 + 0_T_1
    new_state_scale = state_scale * update_scale # (sx*usx,sy*usy,sz*usz)

    new_T = np.eye(4,4, dtype=np.float32)
    new_T[:3,:3] = new_state_rot  # (3,3)
    new_T[:3,3] = new_state_trans # (3,1)
    new_T[:3,0] *= new_state_scale[0] # new_sx
    new_T[:3,1] *= new_state_scale[1] # new_sy
    new_T[:3,2] *= new_state_scale[2] # new_sz

    return new_T


def update_transform(T: np.ndarray,
                     update_t: np.ndarray,
                     update_rot: np.ndarray,
                     update_scale : np.ndarray) -> np.ndarray:
    """Transform行列[4x4]の更新

    Args:
        M (np.ndarray): Transform行列[4x4]
        update_t (np.ndarray): 並進変化[3x1]
        update_rot (np.ndarray): 姿勢(回転行列)変化[3x3]
        update_scale (np.ndarray): スケール変化[3x1]
    Returns:
        np.ndarray: 更新後のTransform行列[4x4]
    """
    
    # クォータニオン(変化量)
    update_quat = rot_to_quat(update_rot)

    return update_transform(T=T, update_t=update_t, update_quat=update_quat, update_scale=update_scale)


def update_transform(T: np.ndarray,
                     update_t: np.ndarray,
                     update_euler_deg: Tuple[float, float, float],
                     euler_state: EulerState,
                     update_scale : np.ndarray
                     ) -> np.ndarray:
    """Transform行列[4x4]の更新

    Args:
        T (np.ndarray): Transform行列[4x4]
        update_t (np.ndarray): 並進変化[3x1]
        update_euler_deg (Tuple[float, float, float]): オイラー角(θ1,θ2,θ3)変化
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: 更新後のTransform行列[4x4]
    """
    # オイラー角から回転行列に変換
    update_theta1_deg, update_theta2_deg, update_theta3_deg = update_euler_deg
    update_rot = euler_to_rot(theta1_deg=update_theta1_deg,
                              theta2_deg=update_theta2_deg,
                              theta3_deg=update_theta3_deg,
                              euler_state=euler_state)
    inv_update_rot = update_rot.T # オイラー角は姿勢(変化)用なので, 座標変換で使うには転置して逆行列にする.
    
    return update_transform(T=T, update_t=update_t, update_rot=inv_update_rot, update_scale=update_scale)

