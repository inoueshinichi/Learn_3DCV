"""座標系
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import numpy as np

from type_hint import *

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