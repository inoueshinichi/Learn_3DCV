"""回転姿勢(クォータニオン)の性質に関するテスト
"""

import os
import sys

module_parent_dir = '/'.join([os.path.dirname(__file__), '..'])
sys.path.append(module_parent_dir)

import math
import inspect

import numpy as np
import scipy as sp

np.set_printoptions(suppress=True) # 指数表記禁止

from BasicModule.quartanion import (
    quat, 
    dot_quat, 
    cat_quat, 
    rotate_points_by_quat, 
    quat_to_rot, 
    norm_quat, 
    normalize_quat, 
    inv_quat
)
from BasicModule.rvec import (
    rvec, 
    rvec_to_rot, 
    rotate_points_by_rvec, 
    rvec_to_quat
)
from BasicModule.rotation import (
    ax_rot, 
    ay_rot, 
    az_rot, 
    rot_to_euler,
    rot_to_rvec,
    rot_to_quat, 
)
from BasicModule.euler import euler_to_rot
from BasicModule.geometry_context import GeometryContext

from type_hint import *
from test_util import test_decorator


def test_move_point_with_quat():
    """クォータニオンによる点群の移動
    """
    # 回転ベクトル
    n = np.array([1,1,1], dtype=np.float32) # x軸
    n /= np.linalg.norm(n)
    theta = math.radians(60)
    rv = rvec(n, theta)
    print("rvec=", rv)

    # 回転行列
    rot = rvec_to_rot(rv)
    print("rot\n", rot)
    print("det(rot)=", np.linalg.det(rot))

    # クォータニオン
    q = quat(n, theta)
    print("q=", q)
    print("norm(q)=", norm_quat(q))

    # 点群
    points = np.random.randn(3,5)
    print("points\n", points)

    # 点群の移動
    moved_points_by_rot = rot @ points # 基準
    moved_points_by_rvec = rotate_points_by_rvec(points, rv) # OK
    moved_points_by_quat = rotate_points_by_quat(points, q) # OK
    

    print("moved_points_by_rot\n", moved_points_by_rot)
    print("moved_points_by_rvec\n", moved_points_by_rvec)
    print("moved_points_by_quat\n", moved_points_by_quat)
    


if __name__ == "__main__":
    test_move_point_with_quat() # OK