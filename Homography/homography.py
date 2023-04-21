"""ホモグラフィ
@note
ある平面から別の平面へ写像する2Dの射影変換.
平面は画像や3D上の平面を指す.
ホモグラフィの用途(例)
1. 画像の位置合わせ
2. 修正
3. テクスチャのワーピング
4. パノラマ作成
etc...

@note
・2Dの同次座標
m' = [x',y',w']^T
m = [x,y,w]
・ホモグラフィ行列
H = [
    [h1, h2, h3],
    [h4, h5, h6],
    [h7, h8, h9]]
・射影変換
m'=Hm

@note 
同次座標の点はスケールと一緒に定義されているので, 
m=[x,y,z]=[αx,αy,αz]=[x/w,y/w,1]は, いずれも同じ2D点を指す.
ホモグラフィ行列Hは, スケールと一緒に定義されるので, 8自由度を持つ.
同次座標の2D点mは, 画像座標x,yと一致させるために点をw=1で正規化する.

@note
点群の扱い方. 点の集合は, [DxN]の配列になる. データは列.
ただし, クラスタリングや分類のための, 特長点の集合は, データを行として扱うことが多い.

@note
ホモグラフィ行列の算出
step1. 2D-2Dの対応点を準備
step2. DLT(Direct Linear Transformation)で同次連立方程式`Ax=0`を準備
step3. DLTの同次連立方程式`Ax=0`をSVD(特異値分解)で最小二乗法`min||Ax||`の最適解を求める.
step4. 最適解のランク整理?
step5. RANSACでノイズ耐性を得る

@note
DLTによる`Ax=0`
x=[h1,h2,h3,h4,h5,h6,h7,h8,h9]^T

※ 1つの対応点につき方程式が2個得られる.
※ Aは対応点の2倍の行を持つ行列になる.
A=[
    [-x1j, -y1j, -1, 0, 0, 0, x1j*x2j, y1j*x2j, x2j], j=1
    [0, 0, 0, -x1j, -y1j, -1, x1j*y2j, y1j*y2j, y2j],
     :
     :
    [-x1n, -y1n, -1, 0, 0, 0, x1n*x2n, y1n*x2n, x2n], j=n
    [0, 0, 0, -x1n, -y1n, -1, x1n*y2n, y1n*y2n, y2n],
]

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

from ransac import Ransac, RansacModel

def homo_normalize(homo_v: np.ndarray) -> np.ndarray:
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

def homo_make(v: np.ndarray) -> np.ndarray:
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

def find_homography(planar1_pts: np.ndarray, planar2_pts: np.ndarray) -> np.ndarray:
    """DLT法とSVDによる最小二乗最適解で, ホモグラフィ行列Hを求める.
       2D座標は同次座標系(x,y,w)

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]

    Returns:
        np.ndarray: ホモグラフィ行列H[3x3]
    """

    if planar1_pts.shape != planar2_pts.shape:
        raise ValueError("Not match number of points between planar1_pts and planar2_ptr. \
                         Given is planar1_pts.shape: {planar1_pts.shape}, \
                         planar2_pts.shape: {planar2_pts_shape}")
    
    ''' 点群の標準化 (数値計算上重要) '''
    eps: float = 1e-9
    # planar1
    m1 = np.mean(planar1_pts[:2], axis=1) # [2x1] (mean_x, mean_y)
    std1 = np.std(planar1_pts[:2], axis=1) # [2x1] (std_x, std_y)
    max_std1 = math.max(std1[0], std1[1]) + eps # max(std_x, std_y)
    C1 = np.diag([1.0 / max_std1, 1.0 / max_std1, 1.0]) # 対角行列
    C1[0,2] = -m1[0] / max_std1
    C1[1,2] = -m1[1] / max_std1
    planar1_pts = C1 @ planar1_pts # 標準化

    # planar2
    m2 = np.mean(planar2_pts[:2], axis=1)
    std2 = np.std(planar2_pts[:2], axis=1)
    max_std2 = math.max(std2[0], std2[1]) + eps
    C2 = np.diag([1.0 / max_std2, 1.0 / max_std2, 1.0])
    C2[0,2] = -m2[0] / max_std2
    C2[1,2] = -m2[1] / max_std2
    planar2_pts = C2 @ planar2_pts # 標準化

    # DLTによる(Ax=0)の係数行列Aを作成
    N: int = planar1_pts.shape[0]
    A = np.zeros((2*N, 9), dtype=np.float32)
    fp, tp = planar1_pts, planar2_pts
    for i in range(N):
        A[2*i] = [ -fp[0,i], -fp[1,i], -1, 0, 0, 0, tp[0,i]*fp[0,i], tp[0,i]*fp[1,i], tp[0,i] ]
        A[2*i+1] = [ 0, 0, 0, -fp[0,i], -fp[1,i], -1, tp[1,i]*fp[0,i], tp[1,i]*fp[1,i], tp[1,i] ]
    
    # 最小二乗法 min|Ax|をSVD(特異値分解)で解く.
    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3,3)) # 最小特異値に対応する直交行列Vの行ベクトルが解.

    # Hは標準化した2D対応点で求めたので, もとに戻す
    H = np.linalg.inv(C2) @ H @ C1

    # h9=1になるように正規化して返す
    return H / H[2,2]


class RansacHomographyModel(RansacModel):
    def __init__(self):
        pass

    @RansacModel.overrides(RansacModel)
    def fit(self, data: np.ndarray) -> Any:
        """ホモグラフィ行列Hを求める

        Args:
            data (np.ndarray): データセット[NxD] N: データ数, D:データ次元数

        Returns:
            Any: ホモグラフィ行列H[3x3]
        """
        pass

    @RansacModel.overrides(RansacModel)
    def get_error(self, data: np.ndarray, estimated_model: Any) -> np.ndarray:
        """データ点数だけ二乗誤差の配列を求める

        Args:
            data (np.ndarray): データセット[NxD] N: データ数, D:データ次元数
            estimated_model (Any): 誤差計算に使用する推定モデル

        Returns:
            np.ndarray: 二乗誤差の配列[Nx1]
        """
        pass


def find_homography_with_ransac(planar1_pts: np.ndarray, 
                                planar2_pts: np.ndarray, 
                                max_iter: int = 1000,
                                match_threshold: int = 10,
                                inlier_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Ransacを用いたホモグラフィ行列Hのロバスト推定

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]
        max_iter (int, optional): Ransacの最大反復回数. Defaults to 1000.
        match_threshold (int, optional): インライア閾値. Defaults to 10.
        inlier_mask (bool, optional): インライアマスクのフラグ. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ロバストモデルH, インライアのマスク
    """
    pass