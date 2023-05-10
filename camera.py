"""カメラ

Projection(Camera)行列 P = K(R|T) [3x4] : 投影変換 + 透視変換
@note カメラ行列Pはスケール不定性がある
P = [
    [p11,p12,p13,p14], = P1^T
    [p21,p22,p23,p24], = P1^T
    [p31,p32,p33,p34]  = P3^T
]

m = [x,y,w], M = [X,Y,Z,W]
s*m = K(P|T) @ M

カメラPose行列 (wRc|wTc) : ワールド座標系Σwからカメラ座標系Σcへのポーズ(姿勢,並進)
カメラView行列 (cRw|cTw) = (R|T)
@note R = wRc^T = cRw, T = -R^T @ T

カメラ内部パラメータ行列 K [3x3] : 上三角行列
カメラ座標系ΣcのZ軸正の向きのZ=1に像面を作る場合, K[2,2] == 1 : OpenCV系(右手座標系Zup)
K = [
    [fx, 1/cosθ, cx], = [fx, s, cx]
    [0, fy/sinθ, cy], = [0, fy, cy]
    [0,  0,  1]]

  = [[1/δx, s cx],
    [0, 1/δy, cy],
    [0, 0, 1]]
    @
    [[f, 0, 0],
     [0, f, 0],
     [0, 0, 1]]

@note Kの各要素の単位はすべて[pixel]

fx = f/δx, fy = f/δy
f : レンズ焦点距離 [mm]
δx : 水平方向撮像素子サイズ [mm/pixel]
δy : 垂直方向撮像素子サイズ [mm/pixel]
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
    """View行列[4x4]からカメラの位置wTc, 姿勢wRc, 各軸方向ベクトル[rx,ry,rz]を求める

    Args:
        V (np.ndarray): View行列[4x4]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: カメラの位置, 姿勢, 各軸方向ベクトル
    """
    if V.shape != (4,4):
        raise ValueError(f"Not match shape (4,4) of view mat. Given is {V.shape}")
    
    # カメラ姿勢
    rot = V[:3,:3].T # 転置必要
    
    # カメラ位置　
    trans = -1.0 * rot @ V[:3,-3]

    # 各軸のベクトル
    forward = geometry_context.forward_axis(rot)
    right = geometry_context.right_axis(rot)
    up = geometry_context.up_axis(rot)

    return (trans, rot, (forward, right, up))


def decomp_camera(self, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """カメラ行列P[3x4]からK[3x3],R[3x3],T[3x1]に分解する
        ・カメラ座標系は, 右手座標系Zupとする(OpenCV準拠).
        視線方向eyeがz軸正の向き.
        x : 右方向
        y : 下方向
        z : 奥行き方向
        eye : z軸正の向き
            eye
           /
          z
         /
        |----x
        |
        y

        ・正規化画像平面(NIP : Normalize Image Plane)は, 
          同次座標系(x,y,w=z)でz=f=1となるxy平面. つまり, (x/z, y/z)
                |
                |
                |
        ------(0,0)-------x
                |
                |
                y

        ・画像平面(画像座標系)は左上が(0,0)であると仮定する
        ・画像平面はz=fとなるxy平面. つまり, (f*x/z + cx, f*y/z + cy)
        ・カメラ中心(cx, cy)
        (0,0)------------x
        |
        |
        |
        y

        この場合, 
        K = [
            [fx>0, s, cx],
            [0, fy>0, cy],
            [0, 0, 1]]]

    Args:
        P (np.ndarray): Camera行列P[3x4]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (K, R, t)
    """
    if P.shape != (3,4):
        raise ValueError(f"Not match shape (3,4) of camera mat. Given is {P.shape}")

    # P[:,:3](3x3)をKとRに分解する (RQ分解: QR分解の派生種(行列の順番が異なるだけ). Rが上三角行列(=K), Q直行行列(=R))
    K, R = np.linalg.rq(P[:,:3]) # Kは上三角行列(カメラ内部パラメータ行列), Rは直行行列(回転行列)

    ''' RQ分解は一意に決まらず, 正か負の2つの解がある.
    https://search.yahoo.co.jp/amp/s/mem-archive.com/2019/04/21/post-1778/%3Famp%3D1%26usqp%3Dmq331AQGsAEggAID
    回転行列Rの行列式を正にしたい(そうしないと, 座標軸が反転する).
    '''

    ''' カメラ行列分解後の処理(必須)
    Step1. T = diag(sign(diag(K)))を作成
        Tの対角成分の符号を確認. (K[2,2]は`1`に正規化しているので正)
        ○ [1] K[0,0]>0, K[1,1]>0, K[2,2]>0  if det(D) > 0 → fx>0,fy>0. カメラ座標Σcの(ax,cy,az)の反転なし. det(R) = +1
        x [2] K[0,0]<0, K[1,1]>0, K[2,2]>0  if det(D) < 0 → fx<0.      カメラ座標Σcのax軸が反転(鏡映状態)   det(R) = -1
        x [3] K[0,0]>0, K[1,1]<0, K[2,2]>0  if det(D) < 0 → fy<0.      カメラ座標Σcのay軸が反転(鏡映状態)   det(R) = -1
        x [4] K[0,0]>0, K[1,1]>0, K[2,2]<0  if det(D) < 0 → z=-1がNIP. カメラ座標Σcのaz軸が反転(鏡映状態)   det(R) = -1
        ○ [5] K[0,0]>0, K[1,1]<0, K[2,2]<0  if det(D) > 0 → fy<0.      カメラ座標Σcの回転姿勢がおかしい.    det(R) = +1
        ○ [6] K[0,0]<0, K[1,1]>0, K[2,2]<0  if det(D) > 0 → fx<0.      カメラ座標Σcの回転姿勢がおかしい.    det(R) = +1
        ○ [7] K[0,0]<0, K[1,1]<0, K[2,2]>0  if det(D) > 0 → fx<0,fy<0. カメラ座標Σcの回転姿勢がおかしい.    det(R) = +1
        x [8] K[0,0]<0, K[1,1]<0, K[2,2]<0  if det(D) < 0 → fx<0,fy<0. z=-1がNIP. カメラ座標Σcの全軸が反転してる(鏡映状態) det(R) = -1

    Step2. Kの対角成分が正になるようにKの符号を反転し, R=[rx,ry,rz]^Tのうち, 反転しているr*の符号を反転する.
        P[:3,:3] = KR = K @ D @ inv(D) @ R. (※ inv(D)=D)
        K = K @ D
        R = D @ R
        if det(D) < 0: D[1,1] *= -1
        Assert 状態[2][3][4]は通常起きない. 発生すれば, カメラ行列Pの作成が間違っている. (間違ってるかも)

    Step3. 内部キャリブレーション行列KのK[2,2]成分を`1`になるようにK全体を正規化する.
    '''

    # Kの対角成分が正になるようにする
    D = np.diag(np.sign(np.diag(K))) # e.g np.dialg([-1,+1,+1]) fx<0

    # 通常は, det(D)>0になるはず. 
    # もしdet(D)<0になるとすれば, カメラ行列構成時に画像平面が左下(0,0)の定義で行った場合, diag(D)=(+1,-1,+1)がありえる.
    if np.linalg.det(D) < 0:      
        if not(D[0,0] < 0 and D[1,1] < 0 and D[2,2] < 0):
            assert False, f"Assert decomposition for camera matrix P. Given is K: {K}"
        D[1,1] *= -1
        
    # マイナス焦点距離とそれに対応する座標軸の符号を反転
    K = K @ D
    R = D @ R # Tはそれ自身が逆行列 TT=I

    # K[2,2]成分で全体を正規化K[2,2]=1
    K /= K[2,2]

    t = np.linalg.inv(K) @ P[:,3] # [3x3]@[3x1]=[3x1]

    return (K, R, t)


def center_from_camera(P: np.ndarray) -> np.ndarrya:
    """Camera行列[3x4]からカメラ中心Cを求める
    カメラ中心は, PC=0 を満たす(PC = K(R|t)C = KRC + Kt = 0).
    なぜなら, C = -R^T*t

    min |PC| -> SVDの最小特異値に対応する右特異ベクトル.
    P = UΣV^T
    C = V^T[-1,:]

    Args:
        P (np.ndarray): Camera行列(射影行列や透視投影行列)[3x4]
    
    Returns:
        np.ndarrya: 3D上の点: カメラ中心[x,y,z,w]
    """
    U,Σ,V = np.linalg.svd(P) # (3,3)@(3,3)@(3,4)
    C = V[-1,:]
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
