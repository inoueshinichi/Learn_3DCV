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

from coordinate_state import (
    CoordinateState,
    CoorRightYupXforwardState, # OpenGL系
    CoorRightZupYforwardState, # OpenCV系
    CoorLeftYupZforwardState,  # DirectX系
    CoorLeftZupXforwardState,  # UnrealEngine系
)

from geometry_context import GeometryContext
from euler_state import EulerState
from Epipolar.fundamental import find_fundamental_with_ransac
from Epipolar.essential import find_essential_with_ransac
from Epipolar.epipolar import compute_epipole

from type_hint import *


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


def decomp_camera(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera行列P[3x4]からK[3x3],R[3x3],T[3x1]に分解する

    Args:
        P (np.ndarray): Camera行列P[3x4]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (K, R, T)
    """
    if P.shape != (3,4):
        raise ValueError(f"Not match shape (3,4) of camera mat. Given is {P.shape}")

    # P[:,:3](3x3)をKとRに分解する (RQ分解: QR分解の派生種(行列の順番が異なるだけ). Rが上三角行列(=K), Q直行行列(=R))
    K, R = np.linalg.rq(P[:,:3]) # Kは上三角行列(カメラ内部パラメータ行列), Rは直行行列(回転行列)

    # K[2,2]成分で全体を正規化K[2,2]=1
    K /= K[2,2]

    '''RQ分解は一意に決まらず, 正か負の2つの解がある.
    https://search.yahoo.co.jp/amp/s/mem-archive.com/2019/04/21/post-1778/%3Famp%3D1%26usqp%3Dmq331AQGsAEggAID
    回転行列Rの行列式を正にしたい(そうしないと, 座標軸が反転する)ので,必要に応じて変換行列Tの符号を反転させている.
    '''
    # Kの対角成分が正になるようにする
    T = np.diag(np.sign(np.diag(K))) # e.g np.dialg([-1,+1,+1]) fx<0
    
    if T[0,0] < 0 and T[1,1] < 0:
        assert(not (T[0,0] < 0 and T[1,1] < 0), f"Assert sign both fx and fy are minus. fx: {K[0,0]}, fy: {K[1,1]}")

    # (t11=-1, t22=-1) : NG
    # (t11=+1, t22=+1) : OK
    # (t11=-1, t22=+1) or (t11=+1, t22=-1) : 補正する
    # 行列式det(T)をチェック. 
    # 行列式は, 1次方程式の可解性(正則行列の有無)の判断, 幾何的には, ±拡大率(マイナスは反転)
    # if np.linalg.det(T) < 0: # det(T) = t11*t22*t33
    #         T[1,1] *= -1 # fyの符号を反転
        
    # マイナス焦点距離とそれに対応する座標軸の符号を反転
    K = K @ T
    R = T @ R # Tはそれ自身が逆行列 TT=I
    T = np.linalg.inv(K) @ P[:,3] # [3x3]@[3x1]=[3x1]

    return (K, R, T)

def center_from_camera(P: np.ndarray) -> np.ndarrya:
    """Camera行列[3x4]からカメラ中心Cを求める

    Args:
        P (np.ndarray): Camera行列(射影行列や透視投影行列)[3x4]
        geome_context (GeometryContext): 座標系定義

    Returns:
        np.ndarrya: 3D上の点: カメラ中心[x,y,z,w]
    """
    K, R, T = decomp_camera(P)

    C = -1.0 * R.T @ T # [3x3][3x1] = [3x1]

    return C


def compute_camera_with_binary_perspective(F: np.ndarray) -> np.ndarray:
    """2視点の場合, 基礎行列Fからカメラ行列Pを復元できる.
    第1視点のカメラ行列をP1=[I|0]とすると,
    第2視点のカメラ行列P2は以下となる.

    P2 = [S[e1|x] @ F | e1 ]

    @warning この行列(P1,P2)を使った三角測量は歪みを生じやすい.
    e.g. 斜めに傾いたような復元結果になったりする.

    e1は, 第一視点のエピ極(e1^T @ F = 0)
    S[e1|x]は左エピ極e1の歪対称行列(交代行列)

    S[e1|x] = [
        [0, -e1_z, e1_y],
        [e1_z, 0, -e1_x],
        [-e1_y, e1_x, 0]]

    Args:
        F (np.ndarray): 基礎行列F [3x3]

    Returns:
        np.ndarray: 第2視点のカメラ行列P2 [3x4]
    """
    if F.shape != (3,3):
        raise ValueError(f"Not match shape (3,3) of fundamental mat. Given is {F.shape}")
    
    # エピ極の計算
    left_e1, _ = compute_epipole(F)

    Se1 = np.ndarray([
        [0, -left_e1[2], left_e1[1]],
        [left_e1[2], 0, -left_e1[0]],
        [-left_e1[1], left_e1[0], 0]
    ], dtype=np.float32)

    M3x3 = Se1 @ F # [3x3]
    return np.hstack((M3x3, left_e1)) # [3x4]


def compute_camera_with_binary_perspective(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2視点の場合, 基本行列Eからカメラ行列Pを復元できる.
    第1視点のカメラ行列をP1=[I|0]とすると,
    第2視点のカメラ行列P2は以下となる.


    基本行列Eから復元されたカメラ行列Pは測量の関係を反映するが, 4つの解が存在する.
    そのうち, 1つだけが2つのカメラの前方にシーンを持つ. 正しい解を選択することは簡単.

    Args:
        E (np.ndarray): 基本行列E [3x3]

    Returns:
        np.ndarray: _description_
    """
    if E.shape != (3,3):
        raise ValueError(f"Not match shape (3,3) of essential mat. Given is {E.shape}")
    
    # Eのランクが2になるようにする
    U,S,V = np.linalg.svd(E) # ノイズあり環境で推定したEはランク落ちしていない可能性があるため
    if np.linalg.det(U @ V) < 0:
        V = -V
    E = U @ np.linalg.diag([1,1,0]) @ V

    # 2種類の行列を作成
    Z = np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,0]
    ], dtype=np.float32) # 交代行列
    W = np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,1]
    ], dtype=np.float32) # inv(W) = W.T

    # 4つの解
    p1 = np.hstack((U @ W @ V, U[:,2]))
    p2 = np.hstack((U @ W @ V, -U[:,2]))
    p3 = np.hstack((U @ W.T @ V, U[:,2]))
    p4 = np.hstack((U @ W.T @ V, -U[:,2]))

    return (p1, p2, p3, p4)
