"""剛体変換

@note 剛体変換
並進 + 回転 + スケーリング(全軸で同じスケール)
2D剛体変換と3D剛体変換が主要.

m' = [x',y',w']
m  = [x,y,w]
H = [
    [a, -b, tx],
    [b, a, ty],
    [h7=0,h8=0,1]]
4自由度.
スケール : s = sqrt(a^2 + b^2)
[[a,-b],[b, a]] = [s*[cos(θ),-sin(θ)],s*[sin(θ),cos(θ)]] = s*R

剛体変換 m' = Hm

b = [x'_1, y'_1, x'_2, y'_2, x'_3, y'_3]^T
x = [a,b,tx,ty]^T

A = [
    [x1, -y1, 1, 0], j = 1
    [y1, x1, 0, 1],
    [x2, -y2, 1, 0], j = 2
    [y2, x2, 0, 1],
    [x3, -y3, 1, 0], j = 3
    [y3, x3, 0, 1]
    
    :

    [xn, -yn, 1, 0], j = n
    [yn, xn, 0, 1],
    ]

連立1次方程式 Ax = b
min ||Ax-b|| を最小二乗法で解く. (LU, QR, SVD...)

"""

import os
import sys
import math

import numpy as np
from scipy import linalg
from scipy import ndimage
# from scipy.misc import imsave, imshow
import cv2
from PIL import Image


from type_hint import *

from BasicModule.ransac import Ransac, RansacModel
from Homography.homography import homo_squared_errors


def find_rigid2D(planar1_pts: np.ndarray, 
                 planar2_pts: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """2Dリジッド行列H(3x3)を求める.
    同次座標(x,y,w)

    Ax=yの特異値分解(SVD)で求める

    Args:
        planar1_pts (np.ndarray): 第一平面の2D点群[3xN]
        planar2_pts (np.ndarray): 第二平面の2D点群[3xN]

    Returns:
        np.ndarray: リジッド行列H (3x3)
    """

    if planar1_pts.shape != planar2_pts.shape:
        raise ValueError("Not match number of points between planar1_pts and planar2_pts. \
                         Given is planar1_pts.shape: {planar1_pts.shape}, \
                         planar2_pts.shape: {planar2_pts_shape}")
    
    if planar1_pts.shape[1] >= 3:
        raise ValueError(f"Number of points planar1_pts and planar2_pts must be >= 3. Given is {planar1_pts.shape[1]}")


    N = planar1_pts.shape[1]

    A = np.zeros((N*2,4), dtype=np.float32)
    b = np.zeros((N*2,1), dtype=np.float32)

    for j in range(0, N):
        # 係数行列 A
        A[2*j,:] = np.array([planar1_pts[0,j], -planar1_pts[1,j], 1, 0], dtype=np.float32)
        A[2*j+1,:] = np.array([planar1_pts[1,j], planar1_pts[0,j], 0, 1], dtype=np.float32)
        # 切片
        b[2*j,-1] = planar2_pts[0,j]
        b[2*j+1,:] = planar2_pts[1,j]

    # ||Ax-b||を最小化する最小二乗解
    a, b, tx, ty = linalg.lstsq(A,b)[0]
    s = math.sqrt(a**2+b**2)
    R = np.array([
        [a/s, -b/s],
        [b/s, a/s]
    ], dtype=np.float32)

    W = np.zeros((3,3), dtype=np.float32)
    W[:2,:2] = s * R
    W[0,2] = tx
    W[1,2] = ty
    W[2,2] = 1

    params = {
        'scale': s,
        'rotation': R,
        'translation': (tx, ty)
    }

    return params, W



def rigid_alignment(faces: Dict[str, np.ndarray],
                    path: str,
                    plot_flag: bool = True,
                    ) -> np.ndarray:
    """画像を位置合わせして, 新たな画像として保存する.
    """

    # 最初の画像の点を参照点とする
    ref_points = faces.values()[0]

    # 各画像を相似変換で変形する
    for face in faces:
        points = faces[face]

        params, W = find_rigid2D(ref_points, points)

        # s: float = params['scale']
        # rot: np.ndarray = params['rotation']
        # tran: List[float] = params['translation']
        inv_W = np.linalg.inv(W)

        img = np.array(Image.open(os.path.join(path, face))) # (H,W,C)
        img_dst = np.zeros_like(img, dtype='uint8')

        # 色チャネル毎に変形する
        for i in range(len(img.shape)):
            
            img_dst[:,:, i] = ndimage.affine_transform(input=img[:,:,:], 
                                                       matrix=inv_W,
                                                       )

        # 表示
        if plot_flag:
            cv2.imshow("aligned-image", img_dst)

        # 境界を切り抜き, 位置合わせした画像を保存する
        h, w = img_dst.shape[:2]
        border = (w+h) / 2
        cv2.imwrite(os.path.join(path, 'aligned/' + face),
                    img_dst[border : h - border, border : w - border],
                    )


