"""クォータニオンに関するAPI
"""

import math
import numpy as np
import scipy as sp
from typing import (
    Tuple, 
    Dict, 
    List, 
    Union,
    Callable,
    TypeVar,
    Generic,
    NoReturn
)

    """実装済み
    delta_quat()
    quat_norm()
    normalize_quat()

    """


def delta_quat(quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """クォータニオンの時間微分

    Arguments:
        quat {np.ndarray} -- 微分を計算するクォータニオン(4x1)
        omega {np.ndarray} -- オブジェクトの角速度[rad/s] (3x1)

    Returns:
        np.ndarray -- dq/dt = - 1/2 * [0, omega] * quat
        (4x1)
    """
    return -0.5 * multiply_quat(np.r_[0, omega], quat)


def quat_norm(quat: np.ndarray) -> float:
    """クォータニオンのノルム

    Arguments:
        quat {np.ndarray} -- (4x1)

    Returns:
        float -- norm
    """
    return np.linalg.norm(quat)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """クォータニオンの正規化

    Arguments:
        quat {np.ndarray} -- (4x1)

    Returns:
        np.ndarray -- (4x1)
    """
    return quat / (quat_norm(quat))


def conj_quat(quat: np.ndarray) -> np.ndarray:
    """共役クォータニオン

    Arguments:
        quat {np.ndarray} -- (4x1)

    Returns:
        np.ndarray -- (4x1)
    """
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def inv_quat(quat: np.ndarray) -> np.ndarray:
    """逆クォータニオン

    Arguments:
        quat {np.ndarray} -- (4x1)

    Returns:
        np.ndarray -- (4x1)
    """
    return conj_quat(quat) / (quat_norm(quat) ** 2)


def multiply_quat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """クォータニオン積(=姿勢状態の回転->別の姿勢状態にする)
    p = (pw, px, py, pz)
    q = (qw, qx, qy, qz)
    クォータニオンpに対して回転クォータニオンqを施して得られる
    クォータニオンは,

    q * p = (
        qw*pw - qx*px - qy*py - qz*pz,
        qx*pw + qw*px - qz*py + qy*pz,
        qy*pw + qz*px + qw*py - qx*pz,
        qz*pw - qy*px + qx*py + qw*pz
    )

    Arguments:
        q {np.ndarray} -- (4x1)
        p {np.ndarray} -- (4x1)

    Returns:
        np.ndarray -- 正規化したクォータニオン積(4x1)
    """
    o1 = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    o2 = q[1]*p[0] + q[0]*p[1] - q[3]*p[2] + q[2]*p[3]
    o3 = q[2]*p[0] + q[3]*p[1] + q[0]*p[2] - q[1]*p[3]
    o4 = q[3]*p[0] - q[2]*p[1] + q[1]*p[2] + q[0]*p[3]

    o = np.array([o1, o2, o3, o4], dtype=np.float32)
    dq = 1 - np.sum(o * o)
    return o * (1 + 0.5 * dq)


