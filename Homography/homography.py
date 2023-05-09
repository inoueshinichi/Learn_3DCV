"""ホモグラフィ
@note ホモグラフィ
ある平面から別の平面へ写像する射影変換(射影行列).
平面は画像(2D)や3D上の平面を指す.
2Dホモグラフィと3Dホモグラフィの2種類が主要.

@note アフィンは長方形を平行四辺形に変形させることしかできないが,
ホモグラフィは台形へ変形させることができる.

@note ホモグラフィの用途(例)
1. 画像の位置合わせ
2. 修正
3. テクスチャのワーピング
4. パノラマ作成
etc...

@note点群の扱い方. 
点の集合は, [DxN]の配列になる. データは列.
ただし, クラスタリングや分類のための, 特長点の集合は, データを行として扱うことが多い.

2Dホモグラフィ行列Hは, スケール不定性(h9で3x3のH全体を正規化しても同じ変換を示す)
を持つので, 8自由度を持つ (h9=1).

@note 2Dホモグラフィ行列H
・2Dの同次座標
m' = [x',y',w']^T
m  = [x,y,w]
H = [
    [h1, h2, h3],
    [h4, h5, h6],
    [h7, h8, h9]]
8自由度
射影変換: m'=Hm 

@note 3Dホモグラフィ行列H
・3Dの同次座標
M' = [x',y',z',w']^T
M  = [x, y, z, w]^T
H = [
    [h1, h2, h3, h4],
    [h5, h6, h7, h8],
    [h9, h10,h11,h12],
    [h13,h14,h15,h16]]
15自由度
射影変換 M'=HM

@note ホモグラフィ行列の算出
step1. 2D-2D(3D-3D)の対応点を準備
step2. DLT(Direct Linear Transformation)で同次連立方程式`Ax=0`を準備
step3. DLTの同次連立方程式`Ax=0`をSVD(特異値分解)で最小二乗法`min||Ax||`の最適解を求める.
step4. 最適解のランク整理?
step5. RANSACでノイズ耐性を得る

@note 2DホモグラフィのDLT法による同次連立方程式(4点アルゴリズム)
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
最低4つの対応点があれば推定できる(ノイズは含まれる)

@note 3DホモグラフィのDLT法による同次連立方程式(5点アルゴリズム)
`Ax=0`
x = [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16]^T

※ 1つの対応点につき方程式が3個得られる.
※ Aは対応点の3倍の行を持つ行列になる.
A=[
    [-x1j*w2j, -y1j*w2j, -z1j*w2j, -w1j*w2j, 0, 0, 0, 0, 0, 0, 0, 0, x1j*x2j, y1j*x2j, z1j*x2j, w1j*x2j], j=1
    [0, 0, 0, 0, -x1j*w2j, -y1j*w2j, -z1j*w2j, -w2j*w1j, 0, 0, 0, 0, x1j*y2j, y1j*y2j, z1j*y2j, w1j*y2j],
    [0, 0, 0, 0, 0, 0, 0, 0, -x1j*w2j, -y1j*w2j, -z1j*w2j, -w1j*w2j, x1j*z2j, y1j*z2j, z1j*z2j, w1j*z2j],
     :
     :
    [-x1n*w2n, -y1n*w2n, -z1n*w2n, -w1n*w2n, 0, 0, 0, 0, 0, 0, 0, 0, x1n*x2n, y1n*x2n, z1n*x2n, w1n*x2n], j=n
    [0, 0, 0, 0, -x1n*w2n, -y1n*w2n, -z1n*w2n, -w2n*w1n, 0, 0, 0, 0, x1n*y2n, y1n*y2n, z1n*y2n, w1n*y2n],
    [0, 0, 0, 0, 0, 0, 0, 0, -x1n*w2n, -y1n*w2n, -z1n*w2n, -w1n*w2n, x1n*z2n, y1n*z2n, z1n*z2n, w1n*z2n],
]
最低5つの対応点があれば推定できる(ノイズは含まれる)

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

from error_cost_func import squared_errors


def homo_squared_errors(x1: np.ndarray, x2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """二乗誤差

    Args:
        x1 (np.ndarray): 第一平面の2D点群[3xN] or [4xN]
        x2 (np.ndarray): 第二平面の2D点群[3xN] or [4xN]
        H (np.ndarray): ホモグラフィ行列 [3x3] or [4x4]

    Returns:
        np.ndarray: 対応点に対応するL2誤差の配列[1xN]
    """
    # ホモグラフィHで第一平面の点群を第2平面の点群に変換
    transformed_x2 = H @ x1 # [3xN]

    # 同次座標系を正規化(w=1)
    transformed_x2 = transformed_x2 / transformed_x2[-1,:] # [3xN] / [1xN] = [3xN] with w = 1

    # 各点の誤差(ユークリッド距離)
    errors = math.sqrt(squared_errors(x2, transformed_x2)) # [1xN] √{ (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 }
    
    return errors


def find_homography2D(planar1_pts: np.ndarray, 
                      planar2_pts: np.ndarray, 
                      eps: float = 1e-9) -> np.ndarray:
    """2Dホモグラフィ行列Hを求める.
       同次座標系(x,y,w)

       4点以上の対応点とDLT法・SVDによる最小二乗最適解.

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]

    Returns:
        np.ndarray: 2Dホモグラフィ行列H[3x3]
    """

    if planar1_pts.shape != planar2_pts.shape:
        raise ValueError("Not match number of points between planar1_pts and planar2_pts. \
                         Given is planar1_pts.shape: {planar1_pts.shape}, \
                         planar2_pts.shape: {planar2_pts_shape}")
    
    if planar1_pts.shape[1] >= 4:
        raise ValueError(f"Number of points planar1_pts and planar2_pts must be >= 4. Given is {planar1_pts.shape[1]}")
    
    ''' 点群の標準化 (数値計算上重要) '''
    # planar1
    m1 = np.mean(planar1_pts[:2], axis=1) # [2x1] (mean_x, mean_y)
    std1 = np.std(planar1_pts[:2], axis=1) # [2x1] (std_x, std_y)
    max_std1 = math.max(std1[0], std1[1]) + eps # max(std_x, std_y)
    C1 = np.diag([1.0 / max_std1, 1.0 / max_std1, 1.0]) # 対角行列 [3x3]
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
    H = V[-1].reshape(3,3) # 最小特異値に対応する直交行列Vの行ベクトルが解.

    # Hは標準化した2D対応点で求めたので, もとに戻す
    H = np.linalg.inv(C2) @ H @ C1

    # h9=1になるように正規化して返す
    return H / H[-1,-1]


class RansacHomography2DModel(RansacModel):
    def __init__(self):
        super(RansacHomography2DModel, self).__init__()

    @RansacModel.overrides(RansacModel)
    def fit(self, data: np.ndarray) -> Any:
        """4つの対応点からホモグラフィ行列Hを求める

        Args:
            data (np.ndarray): データセット[NxD] N: データ数, D:データ次元数
            N = 4
            D = 6 : (x1,y1,w1,x2,y2,w2)
        Returns:
            Any: ホモグラフィ行列H[3x3]
        """
        data = data.T # 転置 [Dx4]
        
        # 第一平面の点群と第二平面の点群に分割
        planar1_pts = data[:3, :] # [3x4]
        planar2_pts = data[3:, :] # [3x4]

        return find_homography2D(planar1_pts, planar2_pts)

    @RansacModel.overrides(RansacModel)
    def get_error(self, data: np.ndarray, estimated_model: Any) -> np.ndarray:
        """データ点数だけ二乗誤差の配列を求める

        Args:
            data (np.ndarray): データセット[NxD] N: データ数, D:データ次元数
            D = 6 : (x1,y1,w1,x2,y2,w2)
            estimated_model (Any): 誤差計算に使用する2D推定ホモグラフィモデル H[3x3] (np.ndarray)

        Returns:
            np.ndarray: 二乗誤差の配列[Nx1]
        """
        data = data.T # 転置 [6xN]

        # 第一平面の点群と第二平面の点群に分割
        planar1_pts = data[:3, :] # [3xN]
        planar2_pts = data[3:, :] # [3xN]

        errors = homo_squared_errors(planar1_pts, planar2_pts, estimated_model)

        return errors


def find_homography2D_with_ransac(planar1_pts: np.ndarray, 
                                  planar2_pts: np.ndarray, 
                                  match_threshold: float = 10,
                                  max_iter: int = 1000) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Ransacを用いた2Dホモグラフィ行列Hのロバスト推定
    同次座標系(x,y,w)

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]
        max_iter (int, optional): Ransacの最大反復回数. Defaults to 1000.
        match_threshold (int, optional): インライア閾値. Defaults to 10.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ロバストモデルH, インライアのマスク
    """
    # RansacHomography2D
    model = RansacHomography2DModel()
    ransac = Ransac(required_min_num_data=4, # 2Dホモグラフィは最低4つの対応点で推定できる.
                    required_inliers=10,
                    match_threshold=match_threshold,
                    max_iter=max_iter)
    
    # 対応点を1つのデータ形式として連結
    data = np.vstack((planar1_pts, planar2_pts)) # [6xN] (x1,y1,w1,x2,y2,w2)
    data = data.T # 転置[Nx6]

    # Ransacによるロバストな推定モデルの取得
    robust_H_2D, inlier_mask = ransac.execute(data=data, model=model)

    return robust_H_2D, inlier_mask


def find_homography3D(planar1_3d_pts: np.ndarray, 
                      planar2_3d_pts: np.ndarray, 
                      eps: float = 1e-9) -> np.ndarray:
    """DLT法とSVDによる最小二乗最適解で, 3Dホモグラフィ行列Hを求める.
    同次座標系(x,y,z,w)

    Args:
        planar1_3d_pts (np.ndarray): 第一共面の3D点群[4xN]
        planar2_3d_pts (np.ndarray): 第二共面の3D点群[4xN]
        
    Returns:
        np.ndarray: 3Dホモグラフィ行列H[4x4]
    """
    if planar1_3d_pts.shape != planar2_3d_pts.shape:
        raise ValueError("Not match number of points between planar1_3d_pts and planar2_3d_pts. \
                         Given is planar1_3d_pts.shape: {planar1_3d_pts.shape}, \
                         planar2_3d_pts.shape: {planar2_3d_pts_shape}")
    
    if planar1_3d_pts.shape[1] >= 4:
        raise ValueError(f"Number of points planar1_3d_pts and planar2_3d_pts must be >= 4. Given is {planar1_3d_pts.shape[1]}")
    

    ''' 点群の標準化 (数値計算上重要) '''
    # planar1
    m1 = np.mean(planar1_3d_pts[:3], axis=1) # [3x1] (mean_x, mean_y, mean_z)
    std1 = np.std(planar1_3d_pts[:3], axis=1) # [3x1] (std_x, std_y, std_z)
    max_std1 = math.max(std1[0], std1[1], std1[2]) + eps # max(std_x, std_y, std_z)
    C1 = np.diag([1.0 / max_std1, 1.0 / max_std1, 1.0 / max_std1, 1.9]) # 対角行列 [4x4]
    C1[0,2] = -m1[0] / max_std1
    C1[1,2] = -m1[1] / max_std1
    C1[2,2] = -m1[2] / max_std1
    planar1_3d_pts = C1 @ planar1_3d_pts # 標準化

    # planar2
    m2 = np.mean(planar2_3d_pts[:3], axis=1)
    std2 = np.std(planar2_3d_pts[:3], axis=1)
    max_std2 = math.max(std2[0], std2[1], std2[2]) + eps
    C2 = np.diag([1.0 / max_std2, 1.0 / max_std2, 1.0 / max_std2, 1.0])
    C2[0,2] = -m2[0] / max_std2
    C2[1,2] = -m2[1] / max_std2
    C2[2,2] = -m2[2] / max_std2
    planar2_3d_pts = C2 @ planar2_3d_pts # 標準化

    # DLTによる(Ax=0)の係数行列Aを作成
    N: int = planar1_3d_pts.shape[0]
    A = np.zeros((3*N, 16), dtype=np.float32)
    fp, tp = planar1_3d_pts, planar2_3d_pts
    for i in range(N):
        A[3*i] = [ -fp[0]*tp[3], -fp[1]*tp[3], -fp[2]*tp[3], -fp[3]*tp[3], 0, 0, 0, 0, 0, 0, 0, 0, fp[0]*tp[0], fp[1]*tp[0], fp[2]*tp[0], fp[3]*tp[0] ]
        A[3*i+1] = [ 0, 0, 0, 0, -fp[0]*tp[3], -fp[1]*tp[3], -fp[2]*tp[3], -fp[3]*tp[3], 0, 0, 0, 0, fp[0]*tp[1], fp[1]*tp[1], fp[2]*tp[1], fp[3]*tp[1] ]
        A[3*i+2] = [ 0, 0, 0, 0, 0, 0, 0, 0, -fp[0]*tp[3], -fp[1]*tp[3], -fp[2]*tp[3], -fp[3]*tp[3], fp[0]*tp[2], fp[1]*tp[2], fp[2]*tp[2], fp[3]*tp[2] ]

    # 最小二乗法 min|Ax|をSVD(特異値分解)で解く.
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(4,4) # 最小特異値に対応する直交行列Vの行ベクトルが解.

    # Hは標準化した2D対応点で求めたので, もとに戻す
    H = np.linalg.inv(C2) @ H @ C1

    # h16=1になるように正規化して返すjum           
    return H / H[-1,-1]
    

class RansacHomography3DModel(RansacModel):
    def __init__(self):
        super(RansacHomography2DModel, self).__init__()

    @RansacModel.overrides(RansacModel)
    def fit(self, data: np.ndarray) -> Any:
        """5つの対応点から3Dホモグラフィ行列Hを求める

        Args:
            data (np.ndarray): データセット[DxN] N: データ数, D:データ次元数
            N = 5
            D = 8 (x1,y1,z1,w1,x2,y2,z2,w2)

        Returns:
            Any: 3Dホモグラフィ行列H[4x4]
        """
        data = data.T # 転置
        
        # 第一平面の点群と第二平面の点群に分割
        planar1_3d_pts = data[:4, :] # [4x5]
        planar2_3d_pts = data[4:, :] # [4x5]

        return find_homography3D(planar1_3d_pts, planar2_3d_pts)

    @RansacModel.overrides(RansacModel)
    def get_error(self, data: np.ndarray, estimated_model: Any) -> np.ndarray:
        """データ点数だけ二乗誤差の配列を求める

        Args:
            data (np.ndarray): データセット[NxD] N: データ数, D:データ次元数
            D = 8 : (x1,y1,z1,w1,x2,y2,z2,w2)
            estimated_model (Any): 誤差計算に使用する3D推定ホモグラフィモデル H[4x4] (np.ndarray)

        Returns:
            np.ndarray: 二乗誤差の配列[Nx1]
        """
        data = data.T # 転置 [8xN]

        # 第一平面の点群と第二平面の点群に分割
        planar1_3d_pts = data[:4, :] # [4xN]
        planar2_3d_pts = data[4:, :] # [4xN]

        errors = homo_squared_errors(planar1_3d_pts, planar2_3d_pts, estimated_model)
        return errors


def find_homography3D_with_ransac(planar1_3d_pts: np.ndarray, 
                                  planar2_3d_pts: np.ndarray, 
                                  match_threshold: float = 10,
                                  max_iter: int = 1000) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Ransacを用いた3Dホモグラフィ行列Hのロバスト推定
    同次座標系(x,y,z,w)

    Args:
        planar1_3d_pts (np.ndarray): 第一平面の2D点群[4xN]
        planar2_3d_pts (np.ndarray): 第二平面の2D点群[4xN]
        match_threshold (float, optional): Ransacの最大反復回数. Defaults to 10.
        max_iter (int, optional): インライア閾値. Defaults to 1000.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ロバストモデルH, インライアのマスク
    """
    # RansacHomography3D
    model = RansacHomography3DModel()
    ransac = Ransac(required_min_num_data=5, # 3Dホモグラフィは最低5つの対応点で推定できる.
                    required_inliers=10,
                    match_threshold=match_threshold,
                    max_iter=max_iter)
    
    # 対応点を1つのデータ形式として連結
    data = np.vstack((planar1_3d_pts, planar2_3d_pts)) # [8xN] (x1,y1,z1,w1,x2,y2,z2,w2)
    data = data.T # 転置[Nx8]


    # Ransacによるロバストな推定モデルの取得
    robust_H_3D, inlier_mask = ransac.execute(data=data, model=model)

    return robust_H_3D, inlier_mask

