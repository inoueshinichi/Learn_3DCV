"""3D幾何の並進・回転
"""

import os
import sys

import numpy as np

import rotation as rot
import rvec
import quartanion as quat
import euler

from geometry_context import GeometryContext

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



def transform_mat4x4(euler_deg_theta: Tuple[float, float, float], 
                     trans: Tuple[float, float, float],
                     geometry_context: GeometryContext) -> np.ndarray:
    """オイラー角と位置ベクトルから4x4の行列を作成

    Args:
        euler_deg_theta1 (float): オイラー角 第1軸回りの回転角 [°]
        euler_deg_theta2 (float): オイラー角 第2軸回りの回転角 [°]
        euler_deg_theta3 (float): オイラー角 第2軸回りの回転角 [°]
        tx (float): _description_
        ty (float): _description_
        tz (float): _description_

    Returns:
        np.ndarray: _description_
    """
    pass

def look_at(target_pos: np.ndarray, 
            camera_pos: np.ndarray, 
            geometry_context: GeometryContext,
            up_axis: Tuple[float, float, float]) -> np.ndarray:
    """カメラのView行列[4x4]を求める

    Args:
        target_pos (np.ndarray): ターゲットの位置ベクトル
        camera_pos (np.ndarray): カメラ中心の位置ベクトル
        up (Tuple[int,int,int], optional): カメラの上向きベクトル. Defaults to [0,1,0].
        デフォルトはY軸正方向.

    Returns:
        np.ndarray: カメラのView行列[4x4]
    """
    return geometry_context.look_at(target_pos, camera_pos, up_axis)