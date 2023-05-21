"""基本行列Eの推定

@note 対応点
x1j = [x1j,y1j,w1j]^T z=+1 or -1
x2j = [x2j,y2j,w2j]^T z=+1 or -1

E = [[e1,e2,e3],
     [e4,e5,e6],
     [e7,e8,e9]]
基本行列Eはスケールが不変なので, 8自由度になる.
rank(E) = 2, 2つの0でない特異値が等しい.

エピポーラ拘束 
x2j^T*E*x1j = 0

DLTによる同次連立方程式
`Ax=0`

SVDによる最小二乗法
min |Ax| 解 -> A = UΣV^T の最小特異値に対応するV^Tの列ベクトル

※1対応点に対して1つの方程式が得られる.
x = [e1,e2,e3,e4,e5,e6,e7,e8,e9]
A = [
    [x2j*x1j, x2j*y1j, x2j*w1j, y2j*x1j, y2j*y1j, y2j*w1j, w2j*x1j, w2j*y1j, w2j*w1j] : j = 1
    :
    :
    [x2n*x1n, x2n*y1n, x2n*w1n, y2n*x1n, y2n*y1n, y2n*w1n, w2n*x1n, w2n*y1n, w2n*w1n] : j = n
]
+ RANSAC
"""

import os
import sys
import math

import numpy as np

from type_hint import *

from BasicModule.ransac import Ransac, RansacModel
from fundamental import find_fundamental_with_ransac


def find_essential_with_ransac(img1_npts: np.ndarray, 
                               img2_npts: np.ndarray,
                               match_threashold: float = 1e-6,
                               max_iter: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
     """基本行列Eの推定. 
     アルゴリズムは基礎行列Fを求める正規化8点アルゴリズム+Ransacと同じ.
     2D同次座標系[x,y,w]

     Args:
         img1_npts (np.ndarray): 視点1の正規化画像座標 [3xN]
         img2_npts (np.ndarray): 視点2の正規化画像座標 [3xN]
         match_threashold (float, optional): インライア閾値. Defaults to 1e-6.
         max_iter (int, optional): Ransacの最大反復回数. Defaults to 5000.

        インライア閾値に使用する誤差はサンプソン距離.
        Sampson = diag(x1j^2 @ E @ x2j)^2 / (Ex1j[0])^2 + (Ex1j[1])^2 + (Ex2j[0])^2 + (Ex2j[1])^2

     Returns:
         np.ndarray: ロバストな基本行列E[3x3]
     """
     return find_fundamental_with_ransac(img1_npts, 
                                         img2_npts, 
                                         match_threashold=match_threashold, 
                                         max_iter=max_iter)


