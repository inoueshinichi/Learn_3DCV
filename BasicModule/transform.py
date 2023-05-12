"""3D幾何の並進・回転
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *

from BasicModule.geometry_context import GeometryContext
from BasicModule.rotation import rot_to_quat
from BasicModule.rvec import *
from BasicModule.quartanion import update_quat, rotate_points_by_quat, quat_to_rot
from BasicModule.euler import euler_to_rot


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


def update_transform(M: np.ndarray,
                     local_add_rot: np.ndarray,
                     local_add_trans: np.ndarray) -> np.ndarray:
    """Poseの更新

    Args:
        M (np.ndarray): Pose行列[4x4]
        local_add_rot (np.ndarray): ローカル座標系での姿勢変化
        local_add_trans (np.ndarray): ローカル座標系での並進変化

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """
    state_rot = M[:3,:3]
    state_trans = M[:3,3]

    new_rot = state_rot @ local_add_rot # 0_R_2 = 0_R_1 @ 1_R_2
    new_trans = state_rot @ local_add_trans + state_trans # 0_T_2 = 0_R_1 @ 1_T_2 + 0_T_1

    new_M = np.zeros((4,4), dtype=np.float32)
    new_M[:3,:3] = new_rot  # (3,3)
    new_M[:3,3] = new_trans # (3,1)
    new_M[3,3] = 1.0

    return new_M


def update_transform(state_quat: np.ndarray,
                     state_trans: np.ndarray,
                     local_add_euler_deg: Tuple[float, float, float], 
                     local_add_trans: Tuple[float, float, float],
                     geo_ctx: GeometryContext) -> Tuple[np.ndarray, np.ndarray]:
    """クォータニオン(回転)と並進によるPoseの更新

    Args:
        state_quat (np.ndarray): 回転状態(クォータニオン) [4x1] (qx,qy,qz,qw)
        state_trans (np.ndarray): 並進状態(平行移動)     [3x1] (tx,ty,tz)
        local_add_euler_deg (Tuple[float, float, float]): オイラー角(θ1_deg, θ2_deg, θ3_deg)
        local_add_trans (Tuple[float, float, float]): 並進(tx,ty,tz)
        geo_ctx (GeometryContext): 幾何コンテキスト

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    
    # オイラー角(差分)
    theta1_deg, theta2_deg, theta3_deg = local_add_euler_deg

    # 回転行列(差分)
    add_rot = euler_to_rot(theta1_deg=theta1_deg, 
                           theta2_deg=theta2_deg, 
                           theta3_deg=theta3_deg, 
                           euler_state=geo_ctx.euler_state) # (3,3)
    
    # クォータニオン(差分)
    add_quat = rot_to_quat(add_rot) # (4,1)

    # 回転要素の更新(new_状態 = cur_状態 + 差分)
    new_quat = update_quat(state_quat, add_quat)

    # 並進要素の更新(new_状態 = cur_状態 + 差分)
    new_trans = rotate_points_by_quat(local_add_trans, state_quat) + state_trans 

    return (new_quat, new_trans)


def get_pose(quat: np.ndarray, 
             trans: np.ndarray) -> np.ndarray:
    """クォータニオンと位置ベクトルからPose行列[4x4]を作成

    Args:
        quat (np.ndarray): 単位クォータニオン[4x1] (qx,qy,qz,qw)
        trans (np.ndarray): 並進要素 [3x1] (tx,ty,tz)

    Returns:
        np.ndarray: Pose行列[4x4]
    """
    if trans.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {trans.shape}")

    rot = quat_to_rot(quat) # (3,3)

    M = np.zeros((4,4), dtype=np.float32)
    M[:3,:3] = rot # 回転要素(3,3)
    M[:3,3] = trans # 並進要素(3,1)
    M[3,3] = 1.0

    return M


def get_scale(M: np.ndarray) -> Tuple[float, float, float]:
    """Pose行列[4x4]からスケール(sx,sy,sz)を求める

    Args:
        M (np.ndarray): Pose行列 [4x4]

    Returns:
        Tuple[float, float, float]: (x,y,z)の各軸のスケール
    """
    if M.shape != (4,4):
        raise ValueError(f"Not match shape (4,4) for Pose matrix. Given is {M.shape}")

    # 回転行列    
    rot = M[:3,:3]

    # R = [rx,ry,rz] 行列は列優先表現なので,Rの3つの列ベクトルが(X軸,Y軸,Z軸)のベクトル
    rx = rot[:,0]
    ry = rot[:,1]
    rz = rot[:,2]

    sx = np.linalg.norm(rx) # 第一列ベクトルのノルム
    sy = np.linalg.norm(ry) # 第二列ベクトルのノルム
    sz = np.linalg.norm(rz) # 第三列ベクトルのノルム

    return (sx, sy, sz)


def get_rot(M: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]から回転行列[3x3]を求める

    Args:
        M (np.ndarray): Pose行列[4x4]

    Returns:
        np.ndarray: 回転行列[3x3]
    """
    if M.shape != (4,4):
        raise ValueError(f"Not match shape (4,4). Given is {M.shape}")
    
    # R=[rx,ry,rz]の各軸のスケール(sx,sy,sz)
    sx, sy, sz = get_scale(M)

    # スケールが1以外の場合があるので正規化
    R = M[:3,:3].copy()
    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    return R


def get_trans(M: np.ndarray) -> Tuple[float, float, float]:
    """Pose行列[4x4]から並進(tx,ty,tz)を求める

    Args:
        M (np.ndarray): Pose行列[4x4]

    Returns:
       Tuple[float, float, float]: (x,y,z)の並進成分
    """
    if M.shape != (4,4):
        raise ValueError(f"Not match shape (4,4). Given is {M.shape}")
    
    # 行列は列優先表現
    tx = M[0,3]
    ty = M[1,3]
    tz = M[2,3]

    return (tx, ty, tz)

