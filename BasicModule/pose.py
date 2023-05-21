"""3Dオブジェクトの姿勢と並進
定義 親座標系に対する子座標系のPose Pose行列[4x4]
@note 子座標を親座標に変換する座標変換Transform行列[4x4]と同義
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *

from BasicModule.geometry_context import GeometryContext
from BasicModule.euler_state import EulerState
from BasicModule.rotation import rot_to_quat
from BasicModule.rvec import *
from BasicModule.quartanion import quat, cat_quat, quat_to_rot
from BasicModule.euler import euler_to_rot, euler_to_quat
from BasicModule.utility import near_equal, near_zero


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


def pose(trans: np.ndarray,
         rot: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]を作成

    Args:
        trans (np.ndarray): 並進要素[3x1] (tx,ty,tz)
        rot (np.ndarray): 姿勢を示す回転行列[3x3]

    Returns:
        np.ndarray: Pose行列[4x4]
    """
    if trans.shape[0] != 3:
        raise ValueError(f"Not match shape (3,1) or (3,). Given is {trans.shape}")
    if rot.shape != (3,3):
            raise ValueError(f"Not match shape (3,3). Given is {rot.shape}")
    
    M = np.eye(4,4,dtype=np.float32)
    M[:3,:3] = rot # 回転要素(3,3)
    M[:3,3] = trans # 並進要素(3,1)

    return M


def pose(trans: np.ndarray,
         quat: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]を作成

    Args:
        trans (np.ndarray): 並進要素[3x1] (tx,ty,tz)
        quat (np.ndarray): 姿勢を示す単位クォータニオン[4x1] (qx,qy,qz,qw)
        

    Returns:
        np.ndarray: Pose行列[4x4]
    """
    # クォータニオン -> 回転行列
    rot = quat_to_rot(quat) # (3,3)

    return pose(trans=trans, rot=rot)


def pose(trans: np.ndarray,
         rvec: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]を作成

    Args:
        trans (np.ndarray): 並進要素[3x1] (tx,ty,tz)
        rvec (np.ndarray): 姿勢を示す回転ベクトル[3x1] (vx,vy,vz)

    Returns:
        np.ndarray: Pose行列[4x4]
    """
    # 回転ベクトル -> 回転行列
    rot = rvec_to_rot(rvec)

    return pose(trans=trans, rot=rot)


def pose(trans: np.ndarray,
         euler_deg: Tuple[float, float, float],
         euler_state: EulerState) -> np.ndarray:
    """Pose行列[4x4]を作成

    Args:
        trans (np.ndarray): 並進要素[3x1] (tx,ty,tz)
        euler_deg (Tuple[float, float, float]): オイラー角(θ1,θ2,θ3)
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: Pose行列[4x4]
    """
    # オイラー角 -> 回転行列
    theta1_deg, theta2_deg, theta3_deg = euler_deg
    rot = euler_to_rot(theta1_deg=theta1_deg,
                       theta2_deg=theta2_deg,
                       theta3_deg=theta3_deg,
                       euler_state=euler_state)
    
    return pose(trans=trans, rot=rot)


def update_pose(M: np.ndarray,
                update_trans: np.ndarray,
                update_quat: np.ndarray) -> np.ndarray:
    """クォータニオンによるPose行列[4x4]の更新

    Args:
        M (np.ndarray): Pose行列[4x4]
        update_trans (np.ndarray): 並進変化[3x1]
        update_quat (np.ndarray): 姿勢(クォータニオン)変化[4x1]
        update_scale (Tuple[float, float, float]): スケール変化[3x1]

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """
    # 現在の状態
    state_scale = get_scale(M)
    state_rot = get_rot(M)
    state_trans = get_trans(M)

    sx, sy, sz = state_scale
    if (not near_equal(sx, 1.0)) or (not near_equal(sy, 1.0)) or (not near_equal(sz, 1.0)):
         raise ValueError(f"scale (sx,sy,sz) of M (Pose) must be 1.0. Given is {state_scale}")
    
    # クォータニオン
    state_quat = rot_to_quat(state_rot) # 状態
    
    # 更新
    new_state_quat = update_quat(state_quat, update_quat) # 姿勢はクォータニオンで更新
    new_state_rot = quat_to_rot(new_state_quat) # 0_R_2 = 0_R_1 @ 1_R_2
    new_state_trans = state_rot @ update_trans + state_trans # 0_T_2 = 0_R_1 @ 1_T_2 + 0_T_1
    
    new_M = np.eye(4,4, dtype=np.float32)
    new_M[:3,:3] = new_state_rot  # (3,3)
    new_M[:3,3] = new_state_trans # (3,1)

    return new_M


def update_pose(M: np.ndarray,  
                update_trans: np.ndarray,
                update_rot: np.ndarray) -> np.ndarray:
    """回転行列によるPose行列[4x4]の更新

    Args:
        M (np.ndarray): Pose行列[4x4]
        update_trans (np.ndarray): 並進変化[3x1]
        update_rot (np.ndarray): 姿勢(回転行列)変化[3x3]

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """
    
    # 回転行列 -> クォータニオン(変化量)
    update_quat = rot_to_quat(update_rot)

    return update_pose(M=M, update_trans=update_trans, update_quat=update_quat)


def update_pose(M: np.ndarray,
                update_trans: np.ndarray,
                update_rvec: np.ndarray) -> np.ndarray:
    """回転ベクトルによるPose行列[4x4]の更新

    Args:
        M (np.ndarray): Pose行列[4x4]
        update_trans (np.ndarray): 並進変化[3x1]
        update_rvec (np.ndarray): 姿勢(回転ベクトル)変化[3x3]

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """

    # 回転ベクトル -> クォータニオン(変化量)
    update_quat = rvec_to_quat(update_rvec)

    return update_pose(M=M, update_trans=update_trans, update_quat=update_quat)
     
    
def update_pose(M: np.ndarray,
                update_trans: np.ndarray,
                update_euler_deg: Tuple[float, float, float],
                euler_state: EulerState) -> np.ndarray:
    """オイラー角によるPoseの更新

    Args:
        M (np.ndarray): Pose行列[4x4]
        update_trans (np.ndarray): 並進変化[3x1]
        update_euler_deg (Tuple[float, float, float]): オイラー角(θ1,θ2,θ3)変化
        euler_state (EulerState): オイラー角の定義

    Returns:
        np.ndarray: 更新後のPose行列[4x4]
    """
    # オイラー角から回転行列に変換
    update_theta1_deg, update_theta2_deg, update_theta3_deg = update_euler_deg
    update_rot = euler_to_rot(theta1_deg=update_theta1_deg,
                              theta2_deg=update_theta2_deg,
                              theta3_deg=update_theta3_deg,
                              euler_state=euler_state)
    
    return update_pose(M=M, update_trans=update_trans, update_rot=update_rot)


def define_child_flame_pose(parent_M: np.ndarray,
                            diff_trans: np.ndarray,
                            diff_quat: np.ndarray) -> np.ndarray:
    """親座標系のPose行列[4x4]のに対する子座標系のPose行列[4x4]を定義する

    Args:
        parent_M (np.ndarray): 親座標系のPose行列[4x4]
        diff_trans (np.ndarray): 並進変化[3x1]
        diff_quat (np.ndarray): 姿勢(クォータニオン)変化[4x1]

    Returns:
        np.ndarray: 子座標系のPose行列[4x4]
    """
    return update_pose(M=parent_M, update_trans=diff_trans, update_quat=diff_quat)


def define_child_flame_pose(parent_M: np.ndarray,
                            diff_trans: np.ndarray,
                            diff_rot: np.ndarray,
                            ) -> np.ndarray:
    """親座標系のPose行列[4x4]のに対する子座標系のPose行列[4x4]を定義する

    Args:
        parent_M (np.ndarray): 親座標系のPose行列[4x4]
        diff_trans (np.ndarray): 並進変化[3x1]
        diff_rot (np.ndarray): 姿勢(回転行列)変化[3x3]

    Returns:
        np.ndarray: 子座標系のPose行列[4x4]
    """
    return update_pose(M=parent_M, update_trans=diff_trans, update_rot=diff_rot)
     

def pose_to_transform(M: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]をTransform行列[4x4]に変換

    Args:
        M (np.ndarray): Pose行列[4x4]

    Returns:
        np.ndarray: Transform行列[4x4]
    """
    # Pose行列の状態
    scale = get_scale(M) # (3,1)
    rot = get_rot(M) # (3,3)
    trans = get_trans(M) # (3,1)

    sx,sy,sz = scale
    if (not near_equal(sx, 1.0)) or (not near_equal(sy, 1.0)) or (not near_equal(sz, 1.0)):
         raise ValueError(f"scale (sx,sy,sz) of M (Pose) must be 1.0. Given is {state_scale}")
    
    # 変換
    inv_rot = rot.T
    inv_trans = -1.0 * inv_rot @ trans

    # Transform行列
    T = np.eye(4,4, dtype=np.float32)
    T[:3,:3] = inv_rot
    T[:,3] = inv_trans
    
    return T
    

def get_scale(M: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]からスケール(sx,sy,sz)を求める

    Args:
        M (np.ndarray): Pose行列 [4x4]

    Returns:
        np.ndarray: (x,y,z)の各軸のスケール
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

    return np.array([sx, sy, sz], dtype=np.float32).reshape(-1,1)


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


def get_trans(M: np.ndarray) -> np.ndarray:
    """Pose行列[4x4]から並進(tx,ty,tz)を求める

    Args:
        M (np.ndarray): Pose行列[4x4]

    Returns:
       np.ndarray: (x,y,z)の並進成分
    """
    if M.shape != (4,4):
        raise ValueError(f"Not match shape (4,4). Given is {M.shape}")
    
    # 行列は列優先表現
    tx = M[0,3]
    ty = M[1,3]
    tz = M[2,3]

    return np.array([tx, ty, tz], dtype=np.ndarray).reshape(-1,1)


def pose_head_to_target(target: np.ndarray, 
                        pos: np.ndarray, 
                        state_quat: np.ndarray,
                        geo_cxt: GeometryContext) -> np.ndarray:
    """ターゲットの方向にオブジェクトのローカル座標系の前方を向ける(回転)

    Args:
        target (np.ndarray): ターゲットの位置ベクトル[3x1]
        pos (np.ndarray): ローカル座標系の位置[3x1]
        quat (np.ndarray): ローカル座標系のクォータニオン(姿勢)[4x1]
        geo_cxt (GeometryContext): 幾何定義

    Returns:
        np.ndarray: ローカル座標系のクォータニオン(姿勢)[4x1]
    """

    if target.shape != (3,1) or pos.shape != (3,1):
        raise ValueError(f"Not match shape (3,1). Given is target's shape: {target.shape}, pos's shape: {pos.shape}")
    
    # ターゲット視線ベクトル
    pt = target - pos
    pt /= np.linalg.norm(pt) # [3x1]

    # ローカル座標系の前方ベクトル
    rot = quat_to_rot(state_quat)
    forward = geo_cxt.forward_axis(rot) # [3x1]

    # 角度θを求める
    cos_theta = np.dot(pt,forward)
    theta = math.acos(cos_theta)

    # 方向ベクトルを求める
    n = np.cross(forward, pt) # クロス積 (forwardとptに垂直方向)

    # クォータニオンを作成
    q = quat(n, theta)

    # qでstate_quatを更新
    new_quat = cat_quat(state_quat, q)

    return new_quat
