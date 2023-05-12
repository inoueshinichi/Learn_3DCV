"""線形補完
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math

import numpy as np

from type_hint import *

def lerp(a: Union[np.ndarray, float], b: Union[np.ndarray, float], f: float) -> Union[np.ndarray, float]:
    """線形補完 (ベクトルのパラメータ表現)

    Args:
        a (Union[np.ndarray, float]): 始点ベクトル[3x1] or 始点スカラー
        b (Union[np.ndarray, float]): 終点ベクトル[3x1] or 終点スカラー
        f (float): 内分点比率 0 - 1

    Returns:
        Union[np.ndarray, float]: 補完ベクトル[3x1] or 終点スカラー
    """
    if a.shape != b.shape:
        raise ValueError(f"Not match shape a and b. Given is a's shape: {a.shape}, b's shape: {b.shape}")

    return a + f * (b - a)