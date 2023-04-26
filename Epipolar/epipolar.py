"""エピ極とエピポーラ線

エピポーラ拘束
x1^T @ F @ x2 = 0

エピ極(e1,e2)は下記を満たす.
左側のe1[x1,y1,w1], 右側のエピ極e2[x2,y2,w2]
F @ e2 = 0
F^T @ e1 = 0 (e1^T @ F = 0)

エピ極(e1,e2)はFの零点(Fe=0)になるので, 
SVDでF->U,Σ,Vと分解して, 最小特異値に対応する固有ベクトル.
e = V[-1]
"""
import os
import sys
import math

import numpy as np

from type_hint import *

def compute_epipole(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """基礎行列Fから左側のエピ極e1[x1,y1,w1]と右側のエピ極e2[x2,y2,w2]

    Args:
        F (np.ndarray): 基礎行列 F[3x3]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [左エピ極e1, 右エピ極e2]
    """
    if F.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {F.shape}")
    
    # e2 : Fの零空間(Fe2=0)
    U2,S2,V2 = np.linalg.svd(F)
    right_e2 = V2[-1]
    right_e2 /= right_e2[2] # w=1に正規化

    # e1 : F^Tの零空間(F^Te1=0)
    U1,S1,V1 = np.linalg.svd(F.T)
    left_e1 = V1[-1]
    left_e1 /= left_e1[2] # w=1に正規化

    return left_e1, right_e2

def compute_right_epiline(F: np.ndarray, planar1_pts: np.ndarray) -> np.ndarray:
    """左側画像点群(planar1_pts)から右側エピラインを求める
    同次座標系(x,y,w)

    Args:
        F (np.ndarray): 基礎行列F [3x3]
        planar1_pts (np.ndarray): 左側画像点群[3xN]

    Returns:
        np.ndarray: 右側エピラインの傾きと切片[2xN]
    """

    if F.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {F.shape}")
    
    if planar1_pts.shape[0] != 3:
        raise ValueError(f"Not match 3 dim of 2D homo coordinate point. Given is {planar1_pts.shape[0]}")
    
    # 基礎行列Fの要素
    f11, f12, f13 = F[0,0], F[0,1], F[0,2]
    f21, f22, f23 = F[1,0], F[1,1], F[1,2]
    f31, f32, f33 = F[2,0], F[2,1], F[2,2]

    N = planar1_pts.shape[1] # データ数 (点数)

    right_epiline_coeffs = np.zeros((2,N), dtype=np.float64)
    for i in range(0, N):
        x, y, w = planar1_pts[0,i], planar1_pts[1,i], planar1_pts[2,i]
        a = -1.0 * (f11*x+f21*y+f31*w) / (f12*x+f22*y+f32*w)
        b = -1.0 * (f13*x+f23*y+f33*w) / (f12*x+f22*y+f32*w)
        right_epiline_coeffs[0,i] = a
        right_epiline_coeffs[1,i] = b
    
    return right_epiline_coeffs


def compute_left_epiline(F: np.ndarray, planar2_pts: np.ndarray) -> np.ndarray:
    """右側画像点群(palanar2_pts)から左側エピラインを求める
    同次座標系(x,y,w)

    Args:
        F (np.ndarray): 基礎行列F [3x3]
        planar2_pts (np.ndarray): 右側画像点群[3xN]

    Returns:
        np.ndarray: 左側エピラインの傾きと切片[2xN]
    """

    if F.shape != (3,3):
        raise ValueError(f"Not match shape (3,3). Given is {F.shape}")
    
    if planar2_pts.shape[0] != 3:
        raise ValueError(f"Not match 3 dim of 2D homo coordinate point. Given is {planar2_pts.shape[0]}")
    
    # 基礎行列Fの要素
    f11, f12, f13 = F[0,0], F[0,1], F[0,2]
    f21, f22, f23 = F[1,0], F[1,1], F[1,2]
    f31, f32, f33 = F[2,0], F[2,1], F[2,2]

    N = planar2_pts.shape[1] # データ数 (点数)

    left_epiline_coeffs = np.zeros((2,N), dtype=np.float64)
    for i in range(0, N):
        x, y, w = planar2_pts[0,i], planar2_pts[1,i], planar2_pts[2,i]
        a = -1.0 * (f11*x+f12*y+f13*w) / (f21*x+f22*y+f23*w)
        b = -1.0 * (f31*x+f32*y+f33*w) / (f21*x+f22*y+f23*w)
        left_epiline_coeffs[0,i] = a
        left_epiline_coeffs[1,i] = b

    return left_epiline_coeffs