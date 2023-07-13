"""画像の主成分分析
"""
import os
import sys

from PIL import Image
import numpy as np

from type_hint import *

def pca(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """主成分分析

    Args:
        x (np.ndarray): [Nxd] N:枚数, d:各行が平板化(flatten)した画像の行ベクトル

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 写像行列(重要度順), 分散, 平均
    """
    N, dim = x.shape

    # 画像ベクトルをセンタリング(平均値を引く)
    mean_x = x.mean(axis=0)
    cx = x - mean_x # (N,d)

    if dim > N:
        # 次元数が画像枚数より多い場合(9割該当)
        # PCA - 高次元のときは, コンパクトな裏技を用いる
        M = np.dot(cx, cx.T) # 共分散行列
        e, EV = np.linalg.eigh(M) # 固有値と固有ベクトル
        tmp = np.dot(cx.T, EV).T # (d,N) @ (d,ここがコンパクトな裏技
