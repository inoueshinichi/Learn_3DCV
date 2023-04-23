"""カメラのView行列に関数処理
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

def get_ncs(img_pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """カメラ内部パラメータ行列Kを用いて画像平面上の点を正規化座標に変換
    同次座標系(x,y,w)

    Args:
        img_pts (np.ndarray): 画像平面上の点群[3xN]
        K (np.ndarray): カメラの内部パラメータ行列[3x3]

    Returns:
        np.ndarray: 正規化座標の点群[3xN]
    """
    if img_pts.shape[0] != 3: 
        raise ValueError(f"Not match shape (3,N). Given is {img_pts.shape}")

    n_pts = np.linalg.inv(K) @ img_pts # [3xN]
    n_pts /= n_pts[-1,:] # w=1
    return n_pts

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
        np.ndarray: カメラView行列[4x4]
    """
    return geometry_context.look_at(target_pos, camera_pos, up_axis)


def camera_pose(V: np.ndarray, 
                geometry_context: GeometryContext) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """View行列[4x4]からカメラの位置, 姿勢, 各軸方向ベクトルを求める

    Args:
        V (np.ndarray): View行列[4x4]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: カメラの位置, 姿勢, 各軸方向ベクトル
    """
    if V.shape != (4,4):
        raise ValueError(f"Not match shape (4,4) of view mat. Given is {V.shape}")
    
    # カメラ位置　
    trans = -V[:3,-1]

    # カメラ姿勢
    rot = V[:3,:3].T # 転置必要

    forward = geometry_context.forward_axis(rot)
    right = geometry_context.right_axis(rot)
    up = geometry_context.up_axis(rot)

    return (trans, rot, (forward, right, up))


    
