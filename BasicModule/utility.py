"""線形補完
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *


def near_equal(a: float, b: float, eps : float = 0.001) -> bool:
    """浮動小数点の同値判別
    fabs(fabs(a)-fabs(b)) < eps if true
    """
    if math.fabs(math.fabs(a) - math.fabs(b)) <= eps:
        return True
    else:
        return False


def near_zero(a: float, eps : float = 0.001) -> bool:
    """浮動小数点のゼロ判別
    fabs(a) < esp if true
    """
    return near_equal(a, 0, eps=eps)



"""同次座標系 (Homogeneous[Perspective] Coordinate)
@note 同次座標系
同次座標の点はスケールと一緒に定義されているので, 
M=[x,y,z,w]=[sx,sy,sz,sw]=[x/w,y/w,z/w,1]は, いずれも同じ3D点を表す.
m=[x,y,w]=[sx,sy,sw]=[x/w,y/w,1]は, いずれも同じ2D点を表す.
最終的にユークリッド座標系の(x,y,z)と一致させるために点をw=1で正規化することになる.

@note Perspective Division
3D空間を2D空間に投影する方法. M=[x,y,z,1] -> m = [x/z,y/z,1]
"""
def homo(v: np.ndarray) -> np.ndarray:
    """同次座標を作成

    Args:
        v (np.ndarray): 2Dor3Dの座標[DxN]
        D: 座標系の次元
        N: 座標点の個数

    Returns:
        np.ndarray: 同次座標[(D+1)xN]
    """
    # 2D or 3D
    if v.shape[0] != 2 or v.shape[0] != 3:
        raise ValueError(f"Not match shape (2or3, N). Given is {v.shape}")
    
    N: int = v.shape[1]
    return np.vstack((v, np.ones((1, N))))  


def normalize_homo(homo_v: np.ndarray) -> np.ndarray:
    """2Dor3Dの同次座標の正規化

    Args:
        homo_v (np.ndarray): 2Dor3Dの同次座標[DxN]
        D: 座標系の次元
        N: 座標点の個数

    Returns:
        np.ndarray: 正規化した同次座標[DxN]
    """
    # 2D or 3D homo
    if homo_v.shape[0] != 2 or homo_v.shape[0] != 3:
        raise ValueError(f"Not match shape (2or3, N). Given is {homo_v.shape}")
    
    N: int = homo_v.shape[1]
    return homo_v / homo_v[-1,:].reshape(-1, N) 


def nip(img_pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """カメラ内部パラメータ行列Kを用いて画像平面上の点を正規化画像座標系に変換
    正規化画像座標系(Normalized Image Plane Coordinate)
    同次座標系(x,y,w=1)

    Args:
        img_pts (np.ndarray): 画像平面上の点群[3xN]
        K (np.ndarray): カメラの内部パラメータ行列[3x3]

    Returns:
        np.ndarray: 正規化座標の点群[3xN]
    """
    if img_pts.shape[0] != 3: 
        raise ValueError(f"Not match shape (3,N). Given is {img_pts.shape}")

    npts = np.linalg.inv(K) @ img_pts # [3xN]
    npts /= npts[-1,:] # w=1
    return npts


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


def get_axis_x(rot: np.ndarray) -> np.ndarray:
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


def get_axis_y(rot: np.ndarray) -> np.ndarray:
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


def get_axis_z(rot: np.ndarray) -> np.ndarray:
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
    tilde_omega: np.ndarray = np.zeros((3, 3), dtype=np.float32)
    tilde_omega[0,1] = omega[2]
    tilde_omega[0,2] = -omega[1]
    tilde_omega[1,0] = -omega[2]
    tilde_omega[1,2] = omega[0]
    tilde_omega[2,0] = omega[1]
    tilde_omega[2,1] = -omega[0]

    return tilde_omega @ rot





        