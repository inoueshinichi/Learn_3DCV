"""同次座標系 (Homogeneous Coordinate System)

@note 
同次座標の点はスケールと一緒に定義されているので, 
m=[x,y,z]=[αx,αy,αz]=[x/w,y/w,1]は, いずれも同じ2D点を指す.
ホモグラフィ行列Hは, スケールと一緒に定義されるので, 8自由度を持つ.
同次座標の2D点mは, 画像座標x,yと一致させるために点をw=1で正規化する.
"""
import os
import sys
import math

import numpy as np

def hcs_make(v: np.ndarray) -> np.ndarray:
    """2Dor3D座標から同次座標を作成

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

def hcs_normalize(homo_v: np.ndarray) -> np.ndarray:
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