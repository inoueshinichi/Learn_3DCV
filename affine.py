"""アフィン変換

@note 2Dアフィン変換
2DホモグラフィHの(h7=h8=0)で求まる.
6自由度.

3Dアフィン変換
"""

import os
import sys
import math

import numpy as np
from scipy import ndimage

import rotation
import rvec
import quartanion
import euler

from geometry_context import GeometryContext
from euler_state import EulerState

from type_hint import *

from ransac import Ransac, RansacModel
from Homography.homography import l2_error


def find_affine2D(planar1_pts: np.ndarray,
                  planar2_pts: np.ndarray,
                  eps: float = 1e-9) -> np.ndarray:
    """2Dアフィン行列Wを求める.
    同次座標系(x,y,w)

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]
        eps (float, optional): ゼロ割防止小数. Defaults to 1e-9.

    Returns:
        np.ndarray: 2Dアフィン行列W[3x3]
    """

    if planar1_pts.shape != planar2_pts.shape:
        raise ValueError("Not match number of points between planar1_pts and planar2_pts. \
                         Given is planar1_pts.shape: {planar1_pts.shape}, \
                         planar2_pts.shape: {planar2_pts_shape}")
    
    if planar1_pts.shape[1] >= 3:
        raise ValueError(f"Number of points planar1_pts and planar2_pts must be >= 3. Given is {planar1_pts.shape[1]}")

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

    # 平均が0になるように調整する. 平行移動はなくなる.
    A = np.concatenate((planar1_pts[:2], planar2_pts[:2]), axis=0)
    U,S,V = np.linalg.svd(A.T)

    # Hartley-Zisserman(第2版)p.130に基づき行列B,Cを求める
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = np.concatenate((C @ np.linalg.pinv(B), np.zeros((2,1))), axis=1)
    W = np.vstack((tmp2, [0,0,1]))

    # 調整を元に戻す
    W = np.linalg.inv(C2) @ W @ C1

    return W / W[2,2]


def alpha_for_triangle(planar_pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """正規化された同次座標系のplanar_ptsをもつ三角形について,
    サイズ(H,W)の透明度マップを作成する.

    Args:
        planar_pts (np.ndarray): _description_
        H (int): _description_
        W (int): _description_

    Returns:
        np.ndarray: _description_
    """

    alpha = np.zeros((H,W), dtype=np.float32)
    for i in range(np.min(planar_pts[0]), np.max(planar_pts[0])):
        for j in range(np.min(planar_pts[1]), np.max(planar_pts[1])):
            x = np.linalg.solve(planar_pts, [i,j,1])
            if min(x) > 0: # すべての係数が正の数
                alpha[i,j] = 1
    return alpha

def embed_image_in_image(embed_img: np.ndarray,
                         target_img: np.ndarray,
                         target_pts: np.ndarray) -> np.ndarrray:
    """アフィン変換による画像埋め込み
    矩形画像は2つの三角形に分割して, それぞれでアフィン変換.
    同次座標系(x,y,w)

    Args:
        embed_img (np.ndarray): 埋め込み画像 Mono or RGB
        target_img (np.ndarray): 埋め込み先画像 Mono or RGB
        target_pts (np.ndarray): 埋め込み先画像内の4点 [3x4]

    Returns:
        np.ndarrray: 埋め込み済み画像 Mono or RGB
    """

    if embed_img.shape != target_img.shape:
        raise ValueError(f"Not match shape. Given embed_img: {embed_img.shape}, target_img: {target_img.shape}")

    eH = embed_img.shape[0]
    eW = embed_img.shape[1]
    tH = target_img.shape[0]
    tW = target_img.shape[1]

    # embed_imgに含まれる`0`要素に+1する
    zero_mask = embed_img[embed_img == 0]
    copy_embed_img = np.copy(embed_img)
    copy_embed_img[zero_mask] = 1 # 視覚的に黒と認識する小数を加算

    planar1_pts = np.array([
        [0,0,1], # top-left
        [eW,0,1], # top-right
        [eW,eH,1], # bottom-right
        [0,eH,1]  # bottom-left
    ], dtype=np.float32).T # [3x4]

    # 第1三角形の3点(右上)
    planar1_tri1_pts = planar1_pts[:, [0,1,2]]
    planar2_tri1_pts = target_pts[:, [0,1,2]]

    # アフィン2D行列A1を推定
    A1 = find_affine2D(planar1_tri1_pts, planar2_tri1_pts)

    # アフィン2D変換 with A1
    img1_rt_t = ndimage.affine_transform(copy_embed_img, A1[:2,:2], (A1[0,2],A1[1,2]), (tH,tW))

    # 第1三角形の透明度マップ
    alpha1 = alpha_for_triangle(planar2_tri1_pts, tH, tW)

    img3 = (1-alpha1)*target_img + alpha1*img1_rt_t

    # 第2の三角形の3点(左下)
    planar1_tri2_pts = planar1_pts[:, [2,3,0]]
    planar2_tri2_pts = target_pts[:, [2,3,0]]

    # アフィン2D行列A2を推定
    A2 = find_affine2D(planar1_tri2_pts, planar2_tri2_pts)

    # アフィン2D with A2
    img1_lt_t = ndimage.affine_transform(copy_embed_img, A2[:2,:2], (A2[0,2],A2[1,2]), (tH,tW))

    # 第2三角形の透明度マップ
    alpha2 = alpha_for_triangle(planar2_tri2_pts, tH, tW)

    img4 = (1-alpha2)*img3 + alpha2*img1_lt_t

    return img4