"""損失関数(誤差)

1. L1ノルム誤差
2. L2ノルム誤差
3. サンプソン距離誤差
4. 平均二乗誤差
5. マハラノビス距離

"""

import os
import sys
import math

import numpy as np

from type_hint import *

def aboslute_errors(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """絶対誤差|x1-x2|

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]
        D : xベクトルの次元数
        N : xベクトルの個数

    Returns:
        np.ndarray: 各絶対誤差 [DxN]
    """

    if x1.shape != x2.shape:
        raise ValueError(f"Not match x1 shape and x2 shape. Given is x1: {x1.shape}, x2: {x2.shape}")
    
    errors = math.abs(x1 - x2) # [DxN]

    return errors

def l1_loss(x1: np.ndarray, x2: np.ndarray) -> float:
    """L1損失 mean(sum(|x1-x2|))

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]
        D : xベクトルの次元数
        N : xベクトルの個数

    Returns:
        float: L1損失
    """
    errors = aboslute_errors(x1, x2) # [DxN]
    distance = np.sum(errors, axis=0) # [1xN]
    l1_loss = np.mean(distance)
    return l1_loss


def squared_errors(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """二乗誤差(x1-x2)^2

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]
        D : xベクトルの次元数
        N : xベクトルの個数

    Returns:
        np.ndarray: 各2乗誤差 [DxN]
    """

    if x1.shape != x2.shape:
        raise ValueError(f"Not match x1 shape and x2 shape. Given is x1: {x1.shape}, x2: {x2.shape}")
    
    errors = (x2 - x1) ** 2 # [DxN]

    return errors

def mean_squared_error(x1: np.ndarray, x2: np.ndarray) -> float:
    """平均二乗誤差(MSE: Mean Squared Error)
    mean(sum((x1-x2)^2))

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]
        D : xベクトルの次元数
        N : xベクトルの個数

    Returns:
        float: MSE
    """
    errors = squared_errors(x1,x2) # [DxN]
    distance = np.sum(errors, axis=0) # [1xN]
    mse = np.mean(distance)
    return mse

def root_mean_squared_error(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """二乗平均平方根誤差(RMSE: Root Mean Squared Error)
    sqrt(mean(sum((x1-x2)^2)))

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]

    Returns:
        np.ndarray: RMSE [1xN]
    """
    return math.sqrt(mean_squared_error(x1,x2))


def l2_loss(x1: np.ndarray, x2: np.ndarray) -> float:
    """L2損失 mean(0.5*sum((x1-x2)^2))

    Args:
        x1 (np.ndarray): 変数1 [DxN]
        x2 (np.ndarray): 変数2 [DxN]
        D : xベクトルの次元数
        N : xベクトルの個数

    Returns:
        float: L2損失
    """
    errors = squared_errors(x1,x2) # [DxN]
    distance = 0.5 * np.sum(errors, axis=0) # [1xN]
    l2_loss = np.mean(distance)
    return l2_loss

