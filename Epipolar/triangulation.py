"""三角測量
2組以上のカメラのカメラ行列を用いて3D上の点(x,y,z,w)を復元する.
同次座標系(x,y,w)

P_i [3x4] : 視点iのカメラ行列
m_ij [3x1] : 視点iのj番目の画像上の同次座標点(x,y,w)
λ_ij スカラ : 視点iのj番目の画像点のスケール
M_j [4x1] : 3D空間上のj番目の点. 求めたい解

DLT法 Ax=0 最小二乗法 min |Ax| with SVD
A : (3N, 4+N)
A = [
    [P_1, -m1j,    0,    0, ...]
    [P_2,    0, -m2j,    0, ...]
    [P_3,    0,    0, -m3j, ...]
]
x = [M_j, λ_1j, λ_2j, λ_3j, ...]^T

最適解M_jは, A=U,Σ,V^Tの最小特異値に対応する固有ベクトルx'=V^T[-1]の
先頭から4つの要素x'[:4] = (X,Y,Z,W)
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

def triangulate_with_multi_persvective(num_rastration_points: int,
                                       img_view_pts_list: List[List[np.ndarray]],
                                       P_list: List[np.ndarray]) -> List[np.ndarray]:
    """複数視点三角測量 2D-2Dの対応点から3D空間上の1点を復元する
    同次座標(x,y,w)

    Args:
        num_rastration_points (int): 3D復元する点群の数
        img_view_pts_list (List[List[np.ndarray]]): 複数視点の対応点のリスト [ [3x1,3x1,...], [3x1,3x1,...], ...]
        
        P_list (List[np.ndarray]): 複数視点のカメラ行列[3x4]のリスト [ [3x4], [3x4], ...]

    Returns:
        List[np.ndarray]: 復元した3D点のリスト [[4x1], [4x1], ...]
    """
    N = len(img_view_pts_list) # 視点数
    M = num_rastration_points  # 復元する点数
    for i in range(0, N):
        if len(img_view_pts_list[i]) != M:
            raise ValueError(f"Not match num_rastration_points : {M}. Given is {len(img_view_pts_list[i])} of {i}th view")
        
        if P_list[i].shape != (3,4):
            raise ValueError(f"Not match shape (3,4). Given is {P_list[i].shape}")
    
    A = np.zeros((3*N, 4+N), dtype=np.float32) # DLTの係数行列 (使い回す)
    pt3d_list = []
        
    for j in range(0, M): # 列 (復元したい点群数)

        A[:,:] = 0 # ゼロで初期化
        for i in range(0, N): # 行 (視点数)
            A[3*i:3*i+3,:4] = P_list[i] # カメラ行列

        # 2D対応点
        A[3*i:3*i+3, 4+j] = -1.0 * img_view_pts_list[i][j]

        # 対応点毎に特異値分解を行う
        U,S,V = np.linalg.svd(A)
        # 最小特異値に対応する固有ベクトルの先頭から4つ
        X = V[-1,:4]
        # 同次座標を正規化 w=1
        X /= X[-1]
        
        pt3d_list.append(X)

    return pt3d_list
        
        
def triangulate_with_binary_persvective(img1_pts_list: List[np.ndarray], 
                                        img2_pts_list: List[np.ndarray], 
                                        P1: np.ndarray, 
                                        P2: np.ndarray) -> List[np.ndarray]:
    """2視点三角測量 : 2D-2Dの対応点から3D空間上の1点を復元する
    同次座標(x,y,w)

    Args:
        img1_pts_list (List[np.ndarray]): 視点1における対応点のリスト [[3x1], [3x1], ...]
        img2_pts_list (List[np.ndarray]): 視点2における対応点のリスト [[3x1], [3x1], ...]
        P1 (np.ndarray): 視点1におけるカメラ行列[3x4]
        P2 (np.ndarray): 視点2におけるカメラ行列[3x4]

    Returns:
        List[np.ndarray]: 復元した3D点のリスト [[4x1], [4x1], ...]
    """

    if len(img1_pts_list) != len(img2_pts_list):
        raise ValueError(f"Not match number of 2D pairs for 3D point. Given is len(img1_pts_list): {len(img1_pts_list)}, len(img2_pts_list): {len(img2_pts_list)}")
    
    if P1.shape != (3,4):
        raise ValueError(f"Not match shape (3,4). Given is {P1.shape}")
    if P2.shape != (3,4):
        raise ValueError(f"Not match shape (3,4). Given is {P2.shape}")
        
    M = len(img1_pts_list)
    A = np.zeros((6,6), dtype=np.float32) # DLTの係数行列 (使い回す)
    pt3d_list = []
    for j in range(0, M):
        A[:,:] = 0 # ゼロで初期化
        A[:3,:4] = P1
        A[3:,:4] = P2
        A[:3,4] = -1.0 * img1_pts_list[j]
        A[3:,5] = -1.0 * img2_pts_list[j]

        # 特異値分解
        U,S,V = np.linalg.svd(A)
        # 最小特異値に対応する固有ベクトルの先頭から4つ
        X = V[-1,:4]
        # 同次座標を正規化 w=1
        X /= X[-1]
        
        pt3d_list.append(X)

    return pt3d_list


