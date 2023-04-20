"""3D幾何の並進・回転
"""

import os
import sys
import math

import numpy as np

import rotation
import rvec
import quartanion
import euler

from geometry_context import GeometryContext
from euler_state import EulerState

from type_hint import *

def translate(trans: Tuple[float, float, float]) -> np.ndarray:
    """並進ベクトル

    Args:
        trans (Tuple[float, float, float]): 並進要素(tx,ty,tz)

    Returns:
        np.ndarray: 並進ベクトル[3x1]
    """
    tx, ty, tz = trans
    translation = np.array([tx, ty, tz], dtype=np.float32)
    return translation

def lerp(a: Union[np.ndarray, float], b: Union[np.ndarray, float], f: float) -> Union[np.ndarray, float]:
    """線形補完 (ベクトルのパラメータ表現)

    Args:
        a (Union[np.ndarray, float]): 始点ベクトル[3x1] or 始点スカラー
        b (Union[np.ndarray, float]): 終点ベクトル[3x1] or 終点スカラー
        f (float): 内分点比率 0 - 1

    Returns:
        Union[np.ndarray, float]: 補完ベクトル[3x1] or 終点スカラー
    """
    if a.shape != b.shape:
        raise ValueError(f"Not match shape a and b. Given is a's shape: {a.shape}, b's shape: {b.shape}")

    return a + f * (b - a)

def update_transform(M: np.ndarray,
                     local_delta_rot: np.ndarray,
                     local_delta_trans: np.ndarray) -> np.ndarray:
    """4x4行列の更新

    Args:
        M (np.ndarray): 4x4の行列
        local_delta_rot (np.ndarray): ローカル座標系での姿勢変化
        local_delta_trans (np.ndarray): ローカル座標系での並進変化

    Returns:
        np.ndarray: 4x4の行列
    """
    state_rot = M[:3,:3]
    state_trans = M[:3,3]

    new_rot = state_rot @ local_delta_rot # 0_R_2 = 0_R_1 @ 1_R_2
    new_trans = state_rot @ local_delta_trans + state_trans # 0_T_2 = 0_R_1 @ 1_T_2 + 0_T_1

    new_M = np.zeros((4,4), dtype=np.float32)
    new_M[:3,:3] = new_rot # [3x3]
    new_M[:3,3] = new_trans # [3x1]
    new_M[3,3] = 1.0

    return new_M

def update_transform(state_quat: np.ndarray,
                     state_trans: np.ndarray,
                     local_delta_euler: np.ndarray, 
                     local_delta_trans: np.ndarray,
                     geometry_context: GeometryContext) -> Tuple[np.ndarray, np.ndarray]:
    
    theta1, theta2, theta3 = local_delta_euler # オイラー角
    R = euler.euler_to_rot(theta1, theta2, theta3, 
                           geometry_context.euler_state) # [3x3]
    delta_quat = rotation.rot_to_quat(R) # [4x1]
    new_quat = quartanion.update_quat(state_quat, delta_quat) # 回転要素の更新
    new_trans = quartanion.rot_quat(local_delta_trans, state_quat) + state_trans # 並進要素の更新

    return (new_quat, new_trans)

def get_transform(quat: np.ndarray, 
                  translation: np.ndarray) -> np.ndarray:
    """クォータニオンと位置ベクトルから4x4の行列を作成

    Args:
        quat (np.ndarray): 単位クォータニオン[4x1] (qx,qy,qz,qw)
        translation (np.ndarray): 並進要素 (tx,ty,tz)

    Returns:
        np.ndarray: 4x4の行列
    """
    if translation.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is {translation.shape}")

    R = quartanion.quat_to_rot(quat) #[3x3]

    M = np.zeros((4,4), dtype=np.float32)
    M[:3,:3] = R # 回転要素[3x3]
    M[:3, 3] = translation # 並進要素[3x1]
    M[3,3] = 1.0

    return M


def look_at(target_pos: np.ndarray, 
            camera_pos: np.ndarray, 
            geometry_context: GeometryContext,
            up_axis: Tuple[float, float, float]) -> np.ndarray:
    """カメラのView行列[4x4]を求める

    Args:
        target_pos (np.ndarray): ターゲットの位置ベクトル
        camera_pos (np.ndarray): カメラ中心の位置ベクトル
        geometry_context: 座標系定義
        up (Tuple[float,float,float], optional): カメラの上向きベクトル.
        
    Returns:
        np.ndarray: カメラのView行列[4x4]
    """
    return geometry_context.look_at(target_pos, camera_pos, up_axis)